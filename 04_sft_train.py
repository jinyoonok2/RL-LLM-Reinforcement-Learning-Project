#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for FinQA - Classification/Ranking Mode
Trains a base LLM to select the best candidate answer from a pool.

This replaces the generation-based approach with a classification approach:
- Input: Question + K candidate answers
- Output: Probability distribution over candidates
- Loss: Cross-entropy to select the highest-reward candidate
- Benefits: Much lower VRAM, faster training, better for RL

Usage:
    python 04_sft_train.py --config configs/models/llama-3.2-3b.yaml
    python 04_sft_train.py --config configs/models/llama-3.2-3b.yaml --quick_test
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from utils.common import setup_logging, load_yaml_config, save_manifest, save_json_data

logger = setup_logging()


@dataclass
class SFTConfig:
    """Classification SFT configuration."""
    # Core settings
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_lora: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = None
    
    # Validation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # Paths
    data_dir: str = "datasets/finqa_with_rewards"
    output_dir: str = "outputs/run_001/04_sft"
    
    # Other
    seed: int = 42
    bf16: bool = field(default_factory=lambda: torch.cuda.is_available())
    fp16: bool = False
    skip_validation: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        cfg_dict = load_yaml_config(yaml_path)
        config = cls()
        
        # Map YAML structure to config fields
        config.base_model = cfg_dict.get('model', {}).get('name', config.base_model)
        
        # LoRA
        lora_cfg = cfg_dict.get('lora', {})
        config.use_lora = lora_cfg.get('use_lora', config.use_lora)
        config.lora_r = lora_cfg.get('r', config.lora_r)
        config.lora_alpha = lora_cfg.get('alpha', config.lora_alpha)
        config.lora_dropout = lora_cfg.get('dropout', config.lora_dropout)
        config.lora_target_modules = lora_cfg.get('target_modules', None)
        
        # Training
        train_cfg = cfg_dict.get('training', {})
        for key in ['epochs', 'batch_size', 'learning_rate', 'warmup_steps', 'max_length', 
                    'gradient_accumulation_steps', 'max_grad_norm', 'bf16', 'fp16']:
            if key in train_cfg:
                setattr(config, key, train_cfg[key])
        
        # Validation
        val_cfg = cfg_dict.get('validation', {})
        for key in ['eval_steps', 'save_steps', 'logging_steps']:
            if key in val_cfg:
                setattr(config, key, val_cfg[key])
        
        # Paths
        paths_cfg = cfg_dict.get('paths', {})
        if 'data_dir' in paths_cfg:
            config.data_dir = paths_cfg['data_dir']
        if 'output_dir' in paths_cfg:
            config.output_dir = paths_cfg['output_dir']
        
        return config


class CandidateRankingModel(nn.Module):
    """Model that ranks candidates by encoding each with the LLM and scoring."""
    
    def __init__(self, base_model, num_candidates: int = 8):
        super().__init__()
        self.base_model = base_model
        self.num_candidates = num_candidates
        self.hidden_size = base_model.config.hidden_size
        
        # Get dtype from base_model
        dtype = next(base_model.parameters()).dtype
        
        # Scoring head: maps pooled representation to scalar score
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1, dtype=dtype)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size * num_candidates, seq_len]
            attention_mask: [batch_size * num_candidates, seq_len]
        
        Returns:
            scores: [batch_size, num_candidates]
        """
        batch_size = input_ids.shape[0] // self.num_candidates
        
        # Encode all candidates
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool: use last hidden state at the last attended position
        # Shape: [batch_size * num_candidates, hidden_size]
        last_hidden = outputs.last_hidden_state
        
        # Get the position of last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, sequence_lengths]
        
        # Move pooled to score_head's device if different (for multi-GPU)
        if pooled.device != next(self.score_head.parameters()).device:
            pooled = pooled.to(next(self.score_head.parameters()).device)
        
        # Score each candidate
        # Shape: [batch_size * num_candidates, 1]
        scores = self.score_head(pooled)
        
        # Reshape to [batch_size, num_candidates]
        scores = scores.view(batch_size, self.num_candidates)
        
        return scores


class FinQACandidateDataset(Dataset):
    """Dataset for candidate ranking."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 512, num_candidates: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_candidates = num_candidates
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load data
        logger.info(f"Loading candidate data from {data_file}")
        with open(data_file, 'r') as f:
            self.examples = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(self.examples)} questions with {num_candidates} candidates each")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex['question']
        candidates = ex['candidates'][:self.num_candidates]  # Ensure exactly num_candidates
        
        # Pad if we have fewer candidates
        while len(candidates) < self.num_candidates:
            candidates.append({
                'answer': '',
                'reward': 0.0,
                'is_gold': False
            })
        
        # Build input text for each candidate: "Question: ... Answer: ..."
        input_texts = []
        rewards = []
        
        for cand in candidates:
            answer = cand.get('answer', '')
            program = cand.get('program', '')
            
            # Format: Question + Answer (simple format for classification)
            if isinstance(program, list) and program:
                program_str = str(program[0]) if program else ''
            else:
                program_str = str(program) if program else ''
            
            # Create input text
            if program_str:
                text = f"Question: {question}\nProgram: {program_str}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
            
            input_texts.append(text)
            rewards.append(float(cand.get('reward', 0.0)))
        
        # Tokenize all candidates
        encodings = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find best candidate (highest reward)
        best_idx = int(np.argmax(rewards))
        
        return {
            'input_ids': encodings['input_ids'],  # [num_candidates, seq_len]
            'attention_mask': encodings['attention_mask'],  # [num_candidates, seq_len]
            'labels': torch.tensor(best_idx, dtype=torch.long),  # scalar
            'rewards': torch.tensor(rewards, dtype=torch.float),  # [num_candidates]
            'example_id': ex.get('id', ''),
            'question': question
        }


def collate_fn(batch):
    """Collate batch of candidate rankings."""
    # Stack candidates from all examples in batch
    # batch: list of dicts, each with input_ids [num_candidates, seq_len]
    
    batch_size = len(batch)
    num_candidates = batch[0]['input_ids'].shape[0]
    seq_len = batch[0]['input_ids'].shape[1]
    
    # Flatten: [batch_size, num_candidates, seq_len] -> [batch_size * num_candidates, seq_len]
    input_ids = torch.cat([x['input_ids'] for x in batch], dim=0)
    attention_mask = torch.cat([x['attention_mask'] for x in batch], dim=0)
    
    # Labels and rewards stay as [batch_size]
    labels = torch.stack([x['labels'] for x in batch])
    rewards = torch.stack([x['rewards'] for x in batch])
    
    return {
        'input_ids': input_ids,  # [batch_size * num_candidates, seq_len]
        'attention_mask': attention_mask,  # [batch_size * num_candidates, seq_len]
        'labels': labels,  # [batch_size]
        'rewards': rewards,  # [batch_size, num_candidates]
        'example_id': [x['example_id'] for x in batch],
        'question': [x['question'] for x in batch]
    }


def setup_model(config: SFTConfig, num_candidates: int = 8):
    """Load model with ranking head."""
    logger.info(f"Loading model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Handle pad token
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new pad_token: [PAD]")
    
    # Determine dtype
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    logger.info(f"Using {dtype} for training")
    
    # Check for multi-GPU
    num_gpus = torch.cuda.device_count()
    
    # Load base model (not CausalLM, just the base encoder)
    if num_gpus > 1:
        logger.info(f"🚀 Multi-GPU detected: {num_gpus} GPUs available")
        logger.info("Using device_map='auto' to distribute model across all GPUs")
        base_model = AutoModel.from_pretrained(
            config.base_model,
            torch_dtype=dtype,
            device_map="auto"
        )
    else:
        base_model = AutoModel.from_pretrained(config.base_model, torch_dtype=dtype)
    
    # Apply LoRA to base model
    if config.use_lora:
        logger.info("Applying LoRA adapters")
        target_modules = config.lora_target_modules
        if target_modules is None:
            # Auto-detect based on model architecture
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Llama style
            logger.info(f"Auto-detected LoRA targets: {target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # We're using base model for ranking
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules
        )
        base_model = get_peft_model(base_model, peft_config)
        base_model.print_trainable_parameters()
    
    # Wrap in ranking model
    model = CandidateRankingModel(base_model, num_candidates)
    
    # Only move to device if not using device_map
    if not (num_gpus > 1):
        model = model.to(config.device)
        logger.info(f"Model moved to {config.device}")
    else:
        # When using device_map, score_head needs to be on the same device as base_model's output
        # The base model outputs from its last layer, so find the last device
        if hasattr(base_model, 'hf_device_map'):
            # Get the device of the last layer (where output comes from)
            last_device = list(base_model.hf_device_map.values())[-1]
            model.score_head = model.score_head.to(last_device)
            logger.info(f"Model distributed across {num_gpus} GPUs, score_head on {last_device}")
        else:
            model.score_head = model.score_head.to('cuda:0')
            logger.info(f"Model distributed across {num_gpus} GPUs")
    
    return model, tokenizer


def train_epoch(model, dataloader, optimizer, scheduler, config: SFTConfig, epoch: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Determine input device (first device where model starts)
    if hasattr(model.base_model, 'hf_device_map'):
        # Model is distributed, find first device
        first_device = next(iter(model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to first device (model will handle distribution)
        input_ids = batch['input_ids'].to(first_device)
        attention_mask = batch['attention_mask'].to(first_device)
        labels = batch['labels'].to(first_device)
        
        # Forward pass
        scores = model(input_ids, attention_mask)  # [batch_size, num_candidates]
        
        # Move labels to same device as scores for loss calculation
        if labels.device != scores.device:
            labels = labels.to(scores.device)
        
        # Loss: cross-entropy to select best candidate
        loss = F.cross_entropy(scores, labels)
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item()
        predictions = scores.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = correct / total
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, config):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    reward_gained = 0
    max_possible_reward = 0
    
    # Determine input device (first device where model starts)
    if hasattr(model.base_model, 'hf_device_map'):
        first_device = next(iter(model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(first_device)
            attention_mask = batch['attention_mask'].to(first_device)
            labels = batch['labels'].to(first_device)
            rewards = batch['rewards'].to(first_device)
            
            scores = model(input_ids, attention_mask)
            
            # Move labels and rewards to same device as scores
            if labels.device != scores.device:
                labels = labels.to(scores.device)
                rewards = rewards.to(scores.device)
            
            loss = F.cross_entropy(scores, labels)
            
            total_loss += loss.item()
            predictions = scores.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Calculate reward metrics
            batch_size = predictions.size(0)
            for i in range(batch_size):
                selected_reward = rewards[i, predictions[i]].item()
                best_reward = rewards[i, labels[i]].item()
                
                reward_gained += selected_reward
                max_possible_reward += best_reward
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    avg_reward = reward_gained / total if total > 0 else 0
    reward_ratio = reward_gained / max_possible_reward if max_possible_reward > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'avg_reward': avg_reward,
        'reward_ratio': reward_ratio
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Classification Training for FinQA")
    parser.add_argument("--config", type=str, default="configs/models/llama-3.2-3b.yaml",
                       help="Path to YAML config")
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test with subset")
    parser.add_argument("--skip_validation", action="store_true",
                       help="Skip validation during training")
    args = parser.parse_args()
    
    # Load config
    logger.info(f"📄 Loading config from: {args.config}")
    config = SFTConfig.from_yaml(args.config)
    
    if args.skip_validation:
        config.skip_validation = True
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Print info
    logger.info("="*70)
    logger.info("🚀 FinQA Supervised Fine-Tuning (Classification Mode)")
    logger.info("="*70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Mode: Candidate Ranking/Selection")
    logger.info(f"LoRA: {config.use_lora}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.learning_rate}")
    logger.info("="*70)
    
    # Load datasets
    data_dir = Path(config.data_dir)
    train_file, val_file = data_dir / 'train.jsonl', data_dir / 'val.jsonl'
    
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Data not found in {data_dir}. Run 02_generate_candidates.py and 03_build_rewards.py first!")
        return 1
    
    # Setup model
    model, tokenizer = setup_model(config, num_candidates=8)
    
    # Load datasets
    train_dataset = FinQACandidateDataset(train_file, tokenizer, config.max_length, num_candidates=8)
    val_dataset = FinQACandidateDataset(val_file, tokenizer, config.max_length, num_candidates=8)
    
    # Quick test mode
    if args.quick_test:
        logger.info("🧪 Quick test mode: 100 train, 20 val samples")
        train_dataset.examples = train_dataset.examples[:100]
        val_dataset.examples = val_dataset.examples[:20]
        config.epochs = 2
        config.eval_steps = 50
    
    logger.info(f"✅ Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Avoid multiprocessing issues with tokenizer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0
    best_reward_ratio = 0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting training...")
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation
        if not config.skip_validation:
            val_metrics = validate(model, val_loader, config)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Avg Reward: {val_metrics['avg_reward']:.4f}, "
                       f"Reward Ratio: {val_metrics['reward_ratio']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                best_reward_ratio = val_metrics['reward_ratio']
                
                # Save model
                model.base_model.save_pretrained(output_dir / "best_model")
                tokenizer.save_pretrained(output_dir / "best_model")
                torch.save(model.score_head.state_dict(), output_dir / "best_model" / "score_head.pt")
                logger.info(f"✅ Saved best model (acc={best_accuracy:.4f}, reward_ratio={best_reward_ratio:.4f})")
    
    # Save final model
    model.base_model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    torch.save(model.score_head.state_dict(), output_dir / "final_model" / "score_head.pt")
    
    # Save manifest
    manifest = {
        'config': {
            'base_model': config.base_model,
            'use_lora': config.use_lora,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate
        },
        'results': {
            'best_accuracy': best_accuracy,
            'best_reward_ratio': best_reward_ratio,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc
        }
    }
    
    save_json_data(manifest, str(output_dir / 'training_manifest.json'))
    
    logger.info("\n🎉 Training Complete!")
    logger.info(f"📁 Outputs: {output_dir}")
    logger.info(f"💡 Best Accuracy: {best_accuracy:.4f}")
    logger.info(f"🎯 Best Reward Ratio: {best_reward_ratio:.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
