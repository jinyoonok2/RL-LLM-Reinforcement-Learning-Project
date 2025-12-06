#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Training for FinQA - Classification Mode
Optimizes candidate selection policy with PPO for better reward maximization.

This uses the classification-based approach (matching SFT):
- Input: Question + K candidate answers
- Output: Probability distribution over candidates
- PPO Loss: Optimize selection to maximize reward with KL penalty
- Benefits: Consistent with SFT, multi-GPU support, memory efficient

Usage:
    python 05_train_ppo.py --policy_ckpt outputs/run_001/04_sft/best_model
    python 05_train_ppo.py --config configs/models/llama-3-8b.yaml --algo_config configs/algorithms/ppo.yaml
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
from copy import deepcopy

from utils.common import setup_logging, load_yaml_config, save_manifest, save_json_data

logger = setup_logging()


@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Model paths
    policy_ckpt: str = "outputs/run_001/04_sft/best_model"
    base_model: str = "meta-llama/Llama-3.2-3B"  # Good balance: 2.6x faster than 8B, better quality than 1B
    
    # Data paths
    data_dir: str = "datasets/finqa_with_rewards"
    output_dir: str = "outputs/run_001/05_ppo"
    
    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 8  # For 2 GPUs (use 12-16 for 4 GPUs)
    mini_batch_size: int = 4  # Half of batch size
    gradient_accumulation_steps: int = 1  # No accumulation needed
    ppo_epochs: int = 2  # Reduced from 4 for faster training
    clip_range: float = 0.2
    kl_coef: float = 0.05
    target_kl: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Training
    num_candidates: int = 8
    max_length: int = 256  # Optimized: covers 99.9% of data, 2x speedup
    total_epochs: int = 10  # Reduced from 20 (PPO converges faster)
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_freq: int = 1  # Evaluate every N epochs
    save_freq: int = 2  # Save every N epochs
    logging_steps: int = 50
    
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
        
        # Map YAML to config fields (similar to SFT)
        if 'model' in cfg_dict:
            config.base_model = cfg_dict['model'].get('name', config.base_model)
        
        if 'training' in cfg_dict:
            train_cfg = cfg_dict['training']
            for key in ['learning_rate', 'batch_size', 'gradient_accumulation_steps', 
                       'max_length', 'bf16', 'fp16']:
                if key in train_cfg:
                    setattr(config, key, train_cfg[key])
        
        if 'paths' in cfg_dict:
            paths = cfg_dict['paths']
            if 'data_dir' in paths:
                config.data_dir = paths['data_dir']
            if 'output_dir' in paths:
                config.output_dir = paths['output_dir']
        
        return config


class CandidateRankingModel(nn.Module):
    """Model that ranks candidates (same as SFT)."""
    
    def __init__(self, base_model, num_candidates: int = 8):
        super().__init__()
        self.base_model = base_model
        self.num_candidates = num_candidates
        self.hidden_size = base_model.config.hidden_size
        
        # Get dtype from base_model
        dtype = next(base_model.parameters()).dtype
        
        # Scoring head
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
        last_hidden = outputs.last_hidden_state
        
        # Pool at last non-padding position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, sequence_lengths]
        
        # Move to score_head device if needed (multi-GPU)
        if pooled.device != next(self.score_head.parameters()).device:
            pooled = pooled.to(next(self.score_head.parameters()).device)
        
        # Score each candidate
        scores = self.score_head(pooled)
        scores = scores.view(batch_size, self.num_candidates)
        
        return scores


class FinQACandidateDataset(Dataset):
    """Dataset for PPO training with candidates."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 512, num_candidates: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_candidates = num_candidates
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load data
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
        
        # Pad if needed
        while len(candidates) < self.num_candidates:
            candidates.append({
                'answer': '',
                'reward': 0.0,
                'is_gold': False
            })
        
        # Build input texts
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
        
        # Tokenize
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
    """Collate batch for PPO."""
    # Flatten candidates
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


def compute_ppo_loss(
    logprobs_policy: torch.Tensor,
    logprobs_ref: torch.Tensor,
    rewards: torch.Tensor,
    clip_range: float = 0.2,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO loss for classification.
    
    Args:
        logprobs_policy: [batch_size, num_candidates] log probs from policy
        logprobs_ref: [batch_size, num_candidates] log probs from reference
        rewards: [batch_size, num_candidates] reward for each candidate
        clip_range: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
    
    Returns:
        loss: scalar loss
        info: dict with metrics
    """
    # Move rewards to same device as logprobs (for multi-GPU)
    if rewards.device != logprobs_policy.device:
        rewards = rewards.to(logprobs_policy.device)
    
    # Use rewards as advantages (can be improved with GAE)
    # Normalize rewards across candidates
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    advantages = advantages / (rewards.std(dim=1, keepdim=True) + 1e-8)
    
    # Compute probability ratios: exp(log(π) - log(π_old))
    # For classification, we compare the full distributions
    ratio = torch.exp(logprobs_policy - logprobs_ref)
    
    # Weighted by advantages (reward-based weighting)
    weighted_ratio = ratio * advantages
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    weighted_clipped = clipped_ratio * advantages
    
    # PPO loss: -min(weighted, clipped)
    policy_loss = -torch.min(weighted_ratio, weighted_clipped).mean()
    
    # Entropy bonus (encourages exploration)
    probs_policy = torch.exp(logprobs_policy)
    entropy = -(probs_policy * logprobs_policy).sum(dim=1).mean()
    entropy_loss = -entropy_coef * entropy
    
    # Total loss
    total_loss = policy_loss + entropy_loss
    
    # KL divergence
    kl = (torch.exp(logprobs_ref) * (logprobs_ref - logprobs_policy)).sum(dim=1).mean()
    
    info = {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
        'entropy_loss': entropy_loss.item(),
        'kl': kl.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, info


def setup_models(config: PPOConfig):
    """Load policy and reference models."""
    logger.info(f"Loading tokenizer from {config.policy_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(config.policy_ckpt)
    
    # Handle pad token
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Determine dtype
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    num_gpus = torch.cuda.device_count()
    
    logger.info(f"Loading policy model from {config.policy_ckpt}")
    
    # Check if policy_ckpt has LoRA adapter
    adapter_config_path = Path(config.policy_ckpt) / "adapter_config.json"
    
    if adapter_config_path.exists():
        # Load base model + LoRA adapter
        logger.info("Detected LoRA checkpoint, loading base model + adapter")
        
        if num_gpus > 1:
            logger.info(f"Multi-GPU: {num_gpus} GPUs, using device_map='auto'")
            base_model = AutoModel.from_pretrained(
                config.base_model,
                torch_dtype=dtype,
                device_map="auto",
                use_cache=False  # Disable KV cache for training
            )
            # Load LoRA adapter
            base_model = PeftModel.from_pretrained(base_model, config.policy_ckpt)
            # Enable gradient checkpointing for memory efficiency
            base_model.gradient_checkpointing_enable()
        else:
            base_model = AutoModel.from_pretrained(config.base_model, torch_dtype=dtype)
            base_model = PeftModel.from_pretrained(base_model, config.policy_ckpt)
        
        # Wrap in ranking model
        policy_model = CandidateRankingModel(base_model, config.num_candidates)
        
        # Load score_head weights
        score_head_path = Path(config.policy_ckpt) / "score_head.pt"
        if score_head_path.exists():
            logger.info("Loading score_head weights")
            score_head_state = torch.load(score_head_path, map_location='cpu')
            policy_model.score_head.load_state_dict(score_head_state)
        
        # Position score_head on correct device
        if num_gpus > 1 and hasattr(base_model, 'hf_device_map'):
            last_device = list(base_model.hf_device_map.values())[-1]
            policy_model.score_head = policy_model.score_head.to(last_device)
            logger.info(f"Score_head on {last_device}")
        elif num_gpus <= 1:
            policy_model = policy_model.to(config.device)
        
    else:
        # Load full model checkpoint (not LoRA)
        logger.info("Loading full model checkpoint")
        raise NotImplementedError("Full model loading not yet implemented - use LoRA checkpoint")
    
    # Create reference model (frozen copy)
    logger.info("Creating reference model (frozen copy)")
    
    if num_gpus > 1:
        ref_base_model = AutoModel.from_pretrained(
            config.base_model,
            torch_dtype=dtype,
            device_map="auto",
            use_cache=False
        )
        ref_base_model = PeftModel.from_pretrained(ref_base_model, config.policy_ckpt)
    else:
        ref_base_model = AutoModel.from_pretrained(config.base_model, torch_dtype=dtype)
        ref_base_model = PeftModel.from_pretrained(ref_base_model, config.policy_ckpt)
    
    ref_model = CandidateRankingModel(ref_base_model, config.num_candidates)
    
    # Load score_head for reference
    if score_head_path.exists():
        score_head_state = torch.load(score_head_path, map_location='cpu')
        ref_model.score_head.load_state_dict(score_head_state)
    
    # Position reference model
    if num_gpus > 1 and hasattr(ref_base_model, 'hf_device_map'):
        last_device = list(ref_base_model.hf_device_map.values())[-1]
        ref_model.score_head = ref_model.score_head.to(last_device)
    elif num_gpus <= 1:
        ref_model = ref_model.to(config.device)
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    logger.info("Models loaded successfully")
    return policy_model, ref_model, tokenizer


def train_ppo_epoch(
    policy_model,
    ref_model,
    dataloader,
    optimizer,
    scheduler,
    config: PPOConfig,
    epoch: int
):
    """Train one PPO epoch."""
    policy_model.train()
    ref_model.eval()
    
    total_loss = 0
    total_policy_loss = 0
    total_entropy = 0
    total_kl = 0
    total_reward = 0
    num_batches = 0
    
    # Determine input device
    if hasattr(policy_model.base_model, 'hf_device_map'):
        first_device = next(iter(policy_model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    progress_bar = tqdm(dataloader, desc=f"PPO Epoch {epoch+1}/{config.total_epochs}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(first_device)
        attention_mask = batch['attention_mask'].to(first_device)
        rewards = batch['rewards'].to(first_device)
        
        # PPO update with multiple inner epochs
        for ppo_epoch in range(config.ppo_epochs):
            # Forward pass - policy
            policy_scores = policy_model(input_ids, attention_mask)
            policy_logprobs = F.log_softmax(policy_scores, dim=1)
            
            # Forward pass - reference (frozen)
            with torch.no_grad():
                ref_scores = ref_model(input_ids, attention_mask)
                ref_logprobs = F.log_softmax(ref_scores, dim=1)
            
            # Compute PPO loss
            loss, info = compute_ppo_loss(
                policy_logprobs,
                ref_logprobs,
                rewards,
                clip_range=config.clip_range,
                entropy_coef=config.entropy_coef
            )
            
            # Add KL penalty
            kl = info['kl']
            if kl > config.target_kl * 1.5:
                # Early stop if KL too high
                logger.debug(f"KL {kl:.4f} > target {config.target_kl}, stopping PPO inner loop")
                break
            
            loss_with_kl = loss + config.kl_coef * kl
            
            # Backward
            loss_with_kl.backward()
            
            # Gradient accumulation
            if (ppo_epoch + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track metrics (only from last inner epoch)
            if ppo_epoch == config.ppo_epochs - 1:
                total_loss += info['total_loss']
                total_policy_loss += info['policy_loss']
                total_entropy += info['entropy']
                total_kl += kl
                total_reward += rewards.mean().item()
                num_batches += 1
        
        # Logging
        if (step + 1) % config.logging_steps == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_reward = total_reward / num_batches if num_batches > 0 else 0
            avg_kl = total_kl / num_batches if num_batches > 0 else 0
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'reward': f'{avg_reward:.4f}',
                'kl': f'{avg_kl:.4f}'
            })
    
    # Final optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0,
        'entropy': total_entropy / num_batches if num_batches > 0 else 0,
        'kl': total_kl / num_batches if num_batches > 0 else 0,
        'avg_reward': total_reward / num_batches if num_batches > 0 else 0
    }
    
    return metrics


def evaluate(model, dataloader, config: PPOConfig):
    """Evaluate model on validation set."""
    model.eval()
    
    total_reward = 0
    correct = 0
    total = 0
    
    # Determine input device
    if hasattr(model.base_model, 'hf_device_map'):
        first_device = next(iter(model.base_model.hf_device_map.values()))
    else:
        first_device = config.device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(first_device)
            attention_mask = batch['attention_mask'].to(first_device)
            rewards = batch['rewards'].to(first_device)
            
            # Get scores
            scores = model(input_ids, attention_mask)
            
            # Predict best candidate
            predictions = scores.argmax(dim=1)
            
            # Check if prediction matches highest reward
            best_reward_idx = rewards.argmax(dim=1)
            correct += (predictions == best_reward_idx).sum().item()
            total += predictions.size(0)
            
            # Track average reward of selected candidates
            batch_indices = torch.arange(predictions.size(0), device=predictions.device)
            selected_rewards = rewards[batch_indices, predictions]
            total_reward += selected_rewards.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_reward = total_reward / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'avg_reward': avg_reward
    }


def save_checkpoint(model, tokenizer, output_dir: Path, name: str = "checkpoint"):
    """Save model checkpoint."""
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base model (LoRA adapter)
    if hasattr(model.base_model, 'save_pretrained'):
        model.base_model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved LoRA adapter to {checkpoint_dir}")
    
    # Save score_head separately
    score_head_path = checkpoint_dir / "score_head.pt"
    torch.save(model.score_head.state_dict(), score_head_path)
    logger.info(f"Saved score_head to {score_head_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for FinQA classification")
    parser.add_argument("--policy_ckpt", type=str, required=True,
                       help="Path to SFT checkpoint (policy initialization)")
    parser.add_argument("--config", type=str,
                       help="Path to model config YAML")
    parser.add_argument("--algo_config", type=str,
                       help="Path to PPO algorithm config YAML")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for PPO checkpoints")
    args = parser.parse_args()
    
    # Load config
    if args.algo_config:
        config = PPOConfig.from_yaml(args.algo_config)
    else:
        config = PPOConfig()
    
    # Override with CLI args
    if args.policy_ckpt:
        config.policy_ckpt = args.policy_ckpt
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("="*70)
    logger.info("🚀 PPO Training for FinQA (Classification Mode)")
    logger.info("="*70)
    logger.info(f"Policy checkpoint: {config.policy_ckpt}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"PPO epochs: {config.ppo_epochs}")
    logger.info(f"Clip range: {config.clip_range}")
    logger.info(f"KL coefficient: {config.kl_coef}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("="*70)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Setup models
    policy_model, ref_model, tokenizer = setup_models(config)
    
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
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
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
        train_metrics = train_ppo_epoch(
            policy_model,
            ref_model,
            train_loader,
            optimizer,
            scheduler,
            config,
            epoch
        )
        
        logger.info(f"Epoch {epoch+1}/{config.total_epochs} - "
                   f"Loss: {train_metrics['loss']:.4f}, "
                   f"Reward: {train_metrics['avg_reward']:.4f}, "
                   f"KL: {train_metrics['kl']:.4f}, "
                   f"Entropy: {train_metrics['entropy']:.4f}")
        
        # Evaluate
        if (epoch + 1) % config.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, config)
            logger.info(f"Validation - "
                       f"Accuracy: {val_metrics['accuracy']:.2%}, "
                       f"Avg Reward: {val_metrics['avg_reward']:.4f}")
            
            # Save best model
            if val_metrics['avg_reward'] > best_reward:
                best_reward = val_metrics['avg_reward']
                save_checkpoint(policy_model, tokenizer, output_dir, "best_model")
                logger.info(f"💾 New best model! Reward: {best_reward:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(policy_model, tokenizer, output_dir, f"checkpoint_epoch_{epoch+1}")
    
    # Final save
    save_checkpoint(policy_model, tokenizer, output_dir, "final_model")
    
    logger.info("="*70)
    logger.info("✅ PPO Training Complete!")
    logger.info(f"Best reward: {best_reward:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
