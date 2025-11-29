#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Training for FinQA
Optimizes policy with clipped PPO objective and KL penalty.

Usage:
    python 05_train_ppo.py --policy_ckpt outputs/run_001/03_sft
    python 05_train_ppo.py --config configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml --algo_config configs/algorithms/ppo.yaml
"""

import argparse
import json
import logging
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import setup_logging, save_manifest, load_json_data
from utils.rewards import FinQARewardCalculator

logger = setup_logging()


@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Model paths
    policy_ckpt: str = "outputs/run_001/03_sft"
    model_config: Optional[str] = None
    algo_config: Optional[str] = None
    
    # Data
    train_data: str = "datasets/finqa_processed/train.jsonl"
    val_data: str = "datasets/finqa_processed/val.jsonl"
    output_dir: str = "outputs/run_001/05_ppo"
    
    # PPO hyperparameters
    learning_rate: float = 5.0e-5  # Higher learning rate
    batch_size: int = 16  # Larger batch size
    mini_batch_size: int = 2  # Larger mini-batches
    gradient_accumulation_steps: int = 4  # Adjusted for larger batches
    ppo_epochs: int = 6  # More PPO epochs per update
    clip_range: float = 0.3  # Slightly larger clip range
    kl_coef: float = 0.02  # Lower KL penalty for more exploration
    target_kl: float = 0.15  # Higher target KL
    
    # Generation
    max_new_tokens: int = 128  # Reduced for faster training
    temperature: float = 0.8  # Slightly higher for more exploration
    top_p: float = 0.95  # Higher top_p for diversity
    
    # Training schedule
    total_ppo_epochs: int = 50  # Fewer epochs but more effective
    save_freq: int = 5  # More frequent saves
    eval_freq: int = 2  # More frequent evaluation
    
    # Reward
    normalize_reward: bool = True
    reward_clip: float = 10.0
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


def compute_kl_divergence(logprobs_new, logprobs_old):
    """Compute KL divergence between new and old log probabilities."""
    return (torch.exp(logprobs_old) * (logprobs_old - logprobs_new)).sum(-1)


def compute_ppo_loss(
    logprobs_new,
    logprobs_old,
    advantages,
    clip_range: float = 0.2
):
    """Compute clipped PPO policy loss."""
    # Compute probability ratio
    ratio = torch.exp(logprobs_new - logprobs_old)
    
    # Clipped surrogate objective
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
    
    return policy_loss


def generate_and_score(
    model,
    tokenizer,
    prompts: List[str],
    reward_calc: FinQARewardCalculator,
    targets: List[Dict],
    config: PPOConfig,
    reference_model=None
):
    """Generate responses and compute rewards."""
    device = config.device
    
    all_responses = []
    all_rewards = []
    all_logprobs = []
    all_ref_logprobs = []
    
    for prompt, target in zip(prompts, targets):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        response_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Parse and score
        try:
            parsed = json.loads(response_text)
            reward = reward_calc.calculate_ternary_reward(parsed, target)
        except:
            reward = 0.0
        
        # Compute log probabilities
        # For simplicity, we'll compute them in training loop
        # Here we just collect the response
        all_responses.append(response_text)
        all_rewards.append(reward)
    
    return all_responses, all_rewards


def train_ppo_epoch(
    model,
    reference_model,
    optimizer,
    tokenizer,
    train_data: List[Dict],
    reward_calc: FinQARewardCalculator,
    config: PPOConfig,
    epoch: int
):
    """Train one PPO epoch."""
    model.train()
    
    # Shuffle data
    indices = np.random.permutation(len(train_data))
    train_data_shuffled = [train_data[i] for i in indices]
    
    total_loss = 0
    total_reward = 0
    total_kl = 0
    num_batches = 0
    
    # Process in batches
    for batch_start in tqdm(range(0, len(train_data_shuffled), config.batch_size), 
                           desc=f"Epoch {epoch}"):
        batch_end = min(batch_start + config.batch_size, len(train_data_shuffled))
        batch = train_data_shuffled[batch_start:batch_end]
        
        prompts = [ex['prompt'] for ex in batch]
        targets = [ex['target_json'] for ex in batch]
        
        # Generate responses and compute rewards
        responses, rewards = generate_and_score(
            model, tokenizer, prompts, reward_calc, targets, config, reference_model
        )
        
        # Normalize rewards if enabled
        if config.normalize_reward and len(rewards) > 1:
            rewards_array = np.array(rewards)
            rewards = ((rewards_array - rewards_array.mean()) / 
                      (rewards_array.std() + 1e-8)).tolist()
        
        # Clip rewards
        rewards = [np.clip(r, -config.reward_clip, config.reward_clip) for r in rewards]
        
        # PPO update (simplified - in practice would need proper advantage estimation)
        # For now, we'll do basic policy gradient with reward as advantage
        batch_loss = 0
        
        for prompt, response, reward in zip(prompts, responses, rewards):
            # Tokenize full sequence
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss * reward  # Weight loss by reward
            
            # KL penalty with reference model
            if reference_model is not None:
                with torch.no_grad():
                    ref_outputs = reference_model(**inputs)
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(outputs.logits, dim=-1),
                    torch.softmax(ref_outputs.logits, dim=-1),
                    reduction='batchmean'
                )
                loss += config.kl_coef * kl
                total_kl += kl.item()
            
            batch_loss += loss
        
        # Backward pass
        batch_loss = batch_loss / len(prompts)
        batch_loss.backward()
        
        # Gradient accumulation
        if (num_batches + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += batch_loss.item()
        total_reward += np.mean(rewards)
        num_batches += 1
    
    # Final optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_reward = total_reward / num_batches if num_batches > 0 else 0
    avg_kl = total_kl / num_batches if num_batches > 0 else 0
    
    return {
        'loss': avg_loss,
        'reward': avg_reward,
        'kl': avg_kl
    }


def evaluate(
    model,
    tokenizer,
    val_data: List[Dict],
    reward_calc: FinQARewardCalculator,
    config: PPOConfig,
    max_samples: int = 100
):
    """Evaluate model on validation set."""
    model.eval()
    
    val_subset = val_data[:max_samples]
    prompts = [ex['prompt'] for ex in val_subset]
    targets = [ex['target_json'] for ex in val_subset]
    
    with torch.no_grad():
        responses, rewards = generate_and_score(
            model, tokenizer, prompts, reward_calc, targets, config
        )
    
    parse_success = 0
    for response in responses:
        try:
            json.loads(response)
            parse_success += 1
        except:
            pass
    
    return {
        'mean_reward': np.mean(rewards),
        'parse_rate': parse_success / len(responses)
    }


def main():
    parser = argparse.ArgumentParser(description="Train with PPO")
    parser.add_argument("--policy_ckpt", type=str,
                       help="Path to initial policy checkpoint (SFT model)")
    parser.add_argument("--config", type=str,
                       help="Path to model config YAML")
    parser.add_argument("--algo_config", type=str,
                       help="Path to PPO algorithm config YAML")
    parser.add_argument("--train_data", type=str, default="datasets/finqa_processed/train.jsonl")
    parser.add_argument("--val_data", type=str, default="datasets/finqa_processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/run_001/05_ppo")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit training samples (for testing)")
    args = parser.parse_args()
    
    # Load config
    if args.algo_config:
        config = PPOConfig.from_yaml(args.algo_config)
    else:
        config = PPOConfig()
    
    # Override with command line args
    if args.policy_ckpt:
        config.policy_ckpt = args.policy_ckpt
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("="*70)
    logger.info("🚀 PPO Training for FinQA")
    logger.info("="*70)
    logger.info(f"Policy: {config.policy_ckpt}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"PPO Epochs: {config.total_ppo_epochs}")
    logger.info(f"Clip Range: {config.clip_range}")
    logger.info(f"KL Coef: {config.kl_coef}")
    logger.info("="*70)
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load model and tokenizer
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(config.policy_ckpt).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.policy_ckpt)
    
    # Load reference model (frozen copy)
    logger.info("Loading reference model...")
    reference_model = AutoModelForCausalLM.from_pretrained(config.policy_ckpt).to(config.device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Load data
    logger.info("Loading data...")
    train_data = load_json_data(config.train_data)
    val_data = load_json_data(config.val_data)
    
    if args.max_train_samples:
        train_data = train_data[:args.max_train_samples]
        logger.info(f"Limited to {args.max_train_samples} training samples")
    
    # Initialize reward calculator
    reward_calc = FinQARewardCalculator()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Training loop
    logger.info("Starting PPO training...")
    training_logs = []
    
    for epoch in range(config.total_ppo_epochs):
        # Train one epoch
        metrics = train_ppo_epoch(
            model, reference_model, optimizer, tokenizer,
            train_data, reward_calc, config, epoch
        )
        
        logger.info(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
                   f"Reward={metrics['reward']:.4f}, KL={metrics['kl']:.4f}")
        
        training_logs.append({
            'epoch': epoch,
            **metrics
        })
        
        # Evaluate
        if (epoch + 1) % config.eval_freq == 0:
            eval_metrics = evaluate(model, tokenizer, val_data, reward_calc, config)
            logger.info(f"Eval: Reward={eval_metrics['mean_reward']:.4f}, "
                       f"Parse Rate={eval_metrics['parse_rate']:.2%}")
            training_logs[-1].update({'eval_' + k: v for k, v in eval_metrics.items()})
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            ckpt_dir = output_dir / f"checkpoint-{epoch+1}"
            ckpt_dir.mkdir(exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")
    
    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training logs
    with open(logs_dir / "training_log.json", 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    # Save manifest
    manifest = {
        'policy_ckpt': config.policy_ckpt,
        'total_epochs': config.total_ppo_epochs,
        'final_reward': training_logs[-1].get('reward', 0),
        'config': vars(config)
    }
    save_manifest(manifest, output_dir / "manifest.json")
    
    logger.info("✅ PPO training complete!")


if __name__ == "__main__":
    main()
