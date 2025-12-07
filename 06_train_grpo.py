#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training for FinQA - Classification Mode
Optimizes candidate selection using group-based relative comparisons.

GRPO Key Differences from PPO:
- Groups multiple responses per prompt and compares them relatively
- Uses group baseline (mean/median) instead of individual advantages
- No reference model needed - uses group statistics for normalization
- More sample efficient for comparing multiple candidates

Usage:
    python 06_train_grpo.py --policy_ckpt outputs/run_001/04_sft/best_model
    python 06_train_grpo.py --config configs/models/llama-3-8b.yaml --algo_config configs/algorithms/grpo.yaml
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

from utils.common import setup_logging, load_yaml_config, save_manifest, save_json_data

logger = setup_logging()


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    # Model paths
    policy_ckpt: str = "outputs/run_001/04_sft/best_model"
    base_model: str = "meta-llama/Llama-3.2-3B"
    
    # Data paths
    data_dir: str = "datasets/finqa_with_rewards"
    output_dir: str = "outputs/run_001/06_grpo"
    
    # GRPO hyperparameters
    learning_rate: float = 1.5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    group_size: int = 4  # Number of candidates to compare per group
    num_groups_per_batch: int = 2  # Number of groups in each batch
    
    # Loss computation
    use_batch_bonus: bool = True  # Apply batch-level reward bonus
    group_baseline: str = "mean"  # "mean" or "median" for baseline
    
    # Training
    num_candidates: int = 8
    max_length: int = 256
    total_epochs: int = 100
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_freq: int = 5
    save_freq: int = 10
    logging_steps: int = 50
    
    # Reward normalization
    normalize_reward: bool = True
    reward_clip: float = 10.0
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    bf16: bool = field(default_factory=lambda: torch.cuda.is_available())
    fp16: bool = False
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        cfg_dict = load_yaml_config(yaml_path)
        config = cls()
        
        if 'model' in cfg_dict:
            config.base_model = cfg_dict['model'].get('name', config.base_model)
        
        if 'training' in cfg_dict:
            train_cfg = cfg_dict['training']
            for key in ['learning_rate', 'batch_size', 'gradient_accumulation_steps', 
                       'max_length', 'bf16', 'fp16']:
                if key in train_cfg:
                    setattr(config, key, train_cfg[key])
        
        # GRPO-specific params
        for key in ['group_size', 'num_groups_per_batch', 'use_batch_bonus', 
                   'group_baseline', 'normalize_reward', 'reward_clip']:
            if key in cfg_dict:
                setattr(config, key, cfg_dict[key])
        
        if 'paths' in cfg_dict:
            paths = cfg_dict['paths']
            if 'data_dir' in paths:
                config.data_dir = paths['data_dir']
            if 'output_dir' in paths:
                config.output_dir = paths['output_dir']
        
        return config


class CandidateRankingModel(nn.Module):
    """Model that ranks candidates (same as SFT/PPO)."""
    
    def __init__(self, base_model, num_candidates: int = 8):
        super().__init__()
        self.base_model = base_model
        self.num_candidates = num_candidates
        self.hidden_size = base_model.config.hidden_size
        
        dtype = next(base_model.parameters()).dtype
        
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
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, sequence_lengths]
        
        if pooled.device != next(self.score_head.parameters()).device:
            pooled = pooled.to(next(self.score_head.parameters()).device)
        
        scores = self.score_head(pooled)
        scores = scores.view(batch_size, self.num_candidates)
        
        return scores


class FinQACandidateDataset(Dataset):
    """Dataset for GRPO training with candidates."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 512, num_candidates: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_candidates = num_candidates
        
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            self.examples = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(self.examples)} questions")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex['question']
        candidates = ex['candidates'][:self.num_candidates]
        
        while len(candidates) < self.num_candidates:
            candidates.append({
                'answer': '',
                'reward': 0.0,
                'is_gold': False
            })
        
        input_texts = []
        rewards = []
        
        for cand in candidates:
            answer = cand.get('answer', '')
            program = cand.get('program', '')
            
            if isinstance(program, list) and program:
                program_str = str(program[0]) if program else ''
            else:
                program_str = str(program) if program else ''
            
            if program_str:
                text = f"Question: {question}\nProgram: {program_str}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
            
            input_texts.append(text)
            rewards.append(float(cand.get('reward', 0.0)))
        
        encodings = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'rewards': torch.tensor(rewards, dtype=torch.float),
            'example_id': ex.get('id', ''),
            'question': question
        }


def collate_fn(batch):
    """Collate batch for GRPO."""
    input_ids = torch.cat([x['input_ids'] for x in batch], dim=0)
    attention_mask = torch.cat([x['attention_mask'] for x in batch], dim=0)
    rewards = torch.stack([x['rewards'] for x in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'rewards': rewards,
        'example_id': [x['example_id'] for x in batch],
        'question': [x['question'] for x in batch]
    }


def compute_grpo_loss(
    logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int = 4,
    baseline: str = "mean",
    use_batch_bonus: bool = True
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute GRPO loss using group-based relative optimization.
    
    GRPO optimizes by comparing candidates within groups:
    1. Divide candidates into groups
    2. Compute group baseline (mean/median reward)
    3. Advantage = reward - baseline
    4. Maximize log-prob weighted by advantage
    
    Args:
        logprobs: [batch_size, num_candidates] log probabilities
        rewards: [batch_size, num_candidates] rewards
        group_size: Number of candidates per group
        baseline: "mean" or "median" for group baseline
        use_batch_bonus: Apply batch-level normalization
    
    Returns:
        loss: scalar loss
        info: dict with metrics
    """
    batch_size, num_candidates = logprobs.shape
    
    # Ensure rewards are on same device
    if rewards.device != logprobs.device:
        rewards = rewards.to(logprobs.device)
    
    # Reshape into groups: [batch_size, num_groups, group_size]
    num_groups = num_candidates // group_size
    if num_candidates % group_size != 0:
        # Pad to make even groups
        pad_size = group_size - (num_candidates % group_size)
        logprobs = F.pad(logprobs, (0, pad_size), value=float('-inf'))
        rewards = F.pad(rewards, (0, pad_size), value=0.0)
        num_groups = (num_candidates + pad_size) // group_size
    
    # Reshape
    logprobs_grouped = logprobs.view(batch_size, num_groups, group_size)
    rewards_grouped = rewards.view(batch_size, num_groups, group_size)
    
    # Compute group baseline
    if baseline == "mean":
        group_baseline = rewards_grouped.mean(dim=2, keepdim=True)
    elif baseline == "median":
        group_baseline = rewards_grouped.median(dim=2, keepdim=True)[0]
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    # Compute advantages (relative to group)
    advantages = rewards_grouped - group_baseline
    
    # Optional: Batch-level normalization
    if use_batch_bonus:
        advantages = advantages / (advantages.std() + 1e-8)
    
    # GRPO objective: maximize log-prob weighted by advantage
    # Loss = -E[advantage * log_prob]
    weighted_logprobs = advantages * logprobs_grouped
    loss = -weighted_logprobs.mean()
    
    # Metrics
    avg_reward = rewards.mean().item()
    avg_advantage = advantages.mean().item()
    advantage_std = advantages.std().item()
    
    # Compute how often model prefers higher-reward candidates
    probs = torch.exp(logprobs)
    predicted_best = probs.argmax(dim=1)
    actual_best = rewards.argmax(dim=1)
    accuracy = (predicted_best == actual_best).float().mean().item()
    
    info = {
        'loss': loss.item(),
        'avg_reward': avg_reward,
        'avg_advantage': avg_advantage,
        'advantage_std': advantage_std,
        'accuracy': accuracy
    }
    
    return loss, info


def setup_model(config: GRPOConfig):
    """Load model for GRPO (no reference model needed)."""
    logger.info(f"Loading tokenizer from {config.policy_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(config.policy_ckpt)
    
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    num_gpus = torch.cuda.device_count()
    
    logger.info(f"Loading model from {config.policy_ckpt}")
    
    adapter_config_path = Path(config.policy_ckpt) / "adapter_config.json"
    
    if adapter_config_path.exists():
        logger.info("Detected LoRA checkpoint")
        
        if num_gpus > 1:
            logger.info(f"Multi-GPU: {num_gpus} GPUs, using device_map='auto'")
            base_model = AutoModel.from_pretrained(
                config.base_model,
                torch_dtype=dtype,
                device_map="auto",
                use_cache=False
            )
            base_model = PeftModel.from_pretrained(base_model, config.policy_ckpt)
            base_model.gradient_checkpointing_enable()
        else:
            base_model = AutoModel.from_pretrained(config.base_model, torch_dtype=dtype)
            base_model = PeftModel.from_pretrained(base_model, config.policy_ckpt)
        
        model = CandidateRankingModel(base_model, config.num_candidates)
        
        score_head_path = Path(config.policy_ckpt) / "score_head.pt"
        if score_head_path.exists():
            logger.info("Loading score_head weights")
            score_head_state = torch.load(score_head_path, map_location='cpu')
            model.score_head.load_state_dict(score_head_state)
        
        if num_gpus > 1 and hasattr(base_model, 'hf_device_map'):
            last_device = list(base_model.hf_device_map.values())[-1]
            model.score_head = model.score_head.to(last_device)
            logger.info(f"Score_head on {last_device}")
        elif num_gpus <= 1:
            model = model.to(config.device)
    else:
        raise NotImplementedError("Full model loading not yet implemented - use LoRA checkpoint")
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def train_grpo_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    config: GRPOConfig,
    epoch: int
):
    """Train one GRPO epoch."""
    model.train()
    
    total_loss = 0
    total_reward = 0
    total_accuracy = 0
    num_batches = 0
    
    if hasattr(model.base_model, 'hf_device_map'):
        first_device = next(iter(model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    progress_bar = tqdm(dataloader, desc=f"GRPO Epoch {epoch+1}/{config.total_epochs}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(first_device)
        attention_mask = batch['attention_mask'].to(first_device)
        rewards = batch['rewards'].to(first_device)
        
        # Normalize rewards if requested
        if config.normalize_reward:
            rewards = torch.clamp(rewards, -config.reward_clip, config.reward_clip)
        
        # Forward pass
        scores = model(input_ids, attention_mask)
        logprobs = F.log_softmax(scores, dim=1)
        
        # Compute GRPO loss
        loss, info = compute_grpo_loss(
            logprobs,
            rewards,
            group_size=config.group_size,
            baseline=config.group_baseline,
            use_batch_bonus=config.use_batch_bonus
        )
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += info['loss']
        total_reward += info['avg_reward']
        total_accuracy += info['accuracy']
        num_batches += 1
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            avg_loss = total_loss / num_batches
            avg_reward = total_reward / num_batches
            avg_acc = total_accuracy / num_batches
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'reward': f'{avg_reward:.4f}',
                'acc': f'{avg_acc:.2%}'
            })
    
    # Final optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'avg_reward': total_reward / num_batches if num_batches > 0 else 0,
        'accuracy': total_accuracy / num_batches if num_batches > 0 else 0
    }
    
    return metrics


def evaluate(model, dataloader, config: GRPOConfig):
    """Evaluate model on validation set."""
    model.eval()
    
    total_reward = 0
    correct = 0
    total = 0
    
    if hasattr(model.base_model, 'hf_device_map'):
        first_device = next(iter(model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(first_device)
            attention_mask = batch['attention_mask'].to(first_device)
            rewards = batch['rewards'].to(first_device)
            
            scores = model(input_ids, attention_mask)
            predictions = scores.argmax(dim=1)
            
            best_reward_idx = rewards.argmax(dim=1).to(predictions.device)
            correct += (predictions == best_reward_idx).sum().item()
            total += predictions.size(0)
            
            batch_indices = torch.arange(predictions.size(0), device=predictions.device)
            selected_rewards = rewards.to(predictions.device)[batch_indices, predictions]
            total_reward += selected_rewards.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_reward = total_reward / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'avg_reward': avg_reward
    }


def save_checkpoint(model, tokenizer, output_dir: Path, name: str = "checkpoint", epoch: int = None, best_reward: float = None):
    """Save model checkpoint."""
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model.base_model, 'save_pretrained'):
        model.base_model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved LoRA adapter to {checkpoint_dir}")
    
    score_head_path = checkpoint_dir / "score_head.pt"
    torch.save(model.score_head.state_dict(), score_head_path)
    logger.info(f"Saved score_head to {score_head_path}")
    
    tokenizer.save_pretrained(checkpoint_dir)
    
    if epoch is not None or best_reward is not None:
        state = {
            'epoch': epoch,
            'best_reward': best_reward
        }
        state_path = checkpoint_dir / "training_state.pt"
        torch.save(state, state_path)
        logger.info(f"Saved training state: epoch={epoch}, best_reward={best_reward:.4f}")
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO training for FinQA classification")
    parser.add_argument("--policy_ckpt", type=str, required=True,
                       help="Path to SFT checkpoint (policy initialization)")
    parser.add_argument("--config", type=str,
                       help="Path to model config YAML")
    parser.add_argument("--algo_config", type=str,
                       help="Path to GRPO algorithm config YAML")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for GRPO checkpoints")
    args = parser.parse_args()
    
    # Load config
    if args.algo_config:
        config = GRPOConfig.from_yaml(args.algo_config)
    else:
        config = GRPOConfig()
    
    # Override with CLI args
    if args.policy_ckpt:
        config.policy_ckpt = args.policy_ckpt
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("="*70)
    logger.info("🚀 GRPO Training for FinQA (Classification Mode)")
    logger.info("="*70)
    logger.info(f"Policy checkpoint: {config.policy_ckpt}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Group size: {config.group_size}")
    logger.info(f"Group baseline: {config.group_baseline}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("="*70)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Load datasets
    data_dir = Path(config.data_dir)
    train_dataset = FinQACandidateDataset(
        data_dir / "train.jsonl",
        tokenizer,
        config.max_length,
        config.num_candidates
    )
    val_dataset = FinQACandidateDataset(
        data_dir / "val.jsonl",
        tokenizer,
        config.max_length,
        config.num_candidates
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.total_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_reward = -float('inf')
    
    for epoch in range(config.total_epochs):
        # Train
        train_metrics = train_grpo_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            epoch
        )
        
        logger.info(f"Epoch {epoch+1}/{config.total_epochs} - "
                   f"Loss: {train_metrics['loss']:.4f}, "
                   f"Reward: {train_metrics['avg_reward']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.2%}")
        
        # Evaluate
        if (epoch + 1) % config.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, config)
            logger.info(f"Validation - "
                       f"Accuracy: {val_metrics['accuracy']:.2%}, "
                       f"Avg Reward: {val_metrics['avg_reward']:.4f}")
            
            # Save best model
            if val_metrics['avg_reward'] > best_reward:
                best_reward = val_metrics['avg_reward']
                save_checkpoint(model, tokenizer, output_dir, "best_model", epoch=epoch+1, best_reward=best_reward)
                logger.info(f"💾 New best model! Reward: {best_reward:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(model, tokenizer, output_dir, f"checkpoint_epoch_{epoch+1}", epoch=epoch+1, best_reward=best_reward)
    
    # Final save
    save_checkpoint(model, tokenizer, output_dir, "final_model", epoch=config.total_epochs, best_reward=best_reward)
    
    logger.info("="*70)
    logger.info("✅ GRPO Training Complete!")
    logger.info(f"Best reward: {best_reward:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
