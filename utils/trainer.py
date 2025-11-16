#!/usr/bin/env python3
"""
Shared SFT Trainer Utilities
Provides reusable training logic for supervised fine-tuning.

This module encapsulates the training loop, checkpoint management, and validation.
Can be reused across different training configurations and modules.
"""

import json
import logging
import torch
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

try:
    from transformers import get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    raise ImportError(f"Missing required packages: {e}. Install with: pip install transformers peft torch")

from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class SFTTrainer:
    """
    Trainer for supervised fine-tuning with LoRA support.
    
    Features:
    - Training loop with gradient accumulation
    - Checkpoint saving and loading
    - Validation using ModelEvaluator
    - Training metrics tracking
    - Manifest generation
    """
    
    def __init__(self, model, tokenizer, config, reward_fn=None, collate_fn=None):
        """
        Initialize SFT trainer.
        
        Args:
            model: Pre-loaded model (with or without LoRA)
            tokenizer: Pre-loaded tokenizer
            config: Training configuration object
            reward_fn: Optional reward function for validation
            collate_fn: Optional custom collate function for DataLoader
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.collate_fn = collate_fn
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'ckpt_sft').mkdir(exist_ok=True)
        (self.output_dir / 'valid_samples').mkdir(exist_ok=True)
        
        # Training tracking
        self.global_step = 0
        self.best_val_reward = -float('inf')
        self.train_losses = []
        self.val_rewards = []
        
        # Create evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            reward_fn=self.reward_fn,
            device=config.device,
            max_new_tokens=getattr(config, 'max_new_tokens', 128),
            temperature=getattr(config, 'temperature', 0.7),
            top_p=getattr(config, 'top_p', 0.9)
        )
    
    def train(self, train_dataset, val_dataset):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        logger.info("🚀 Starting SFT training")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn  # Use custom collate if provided
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        total_steps = len(train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.epochs):
            logger.info(f"\n📖 Epoch {epoch + 1}/{self.config.epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Check for NaN/Inf before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {step}, skipping batch")
                    if step < 10:
                        logger.error("Getting NaN in first 10 steps - training cannot proceed!")
                        raise RuntimeError("Training failed with NaN loss in early steps")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        self.train_losses.append(avg_loss)
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                        })
                    
                    # Validation (skip if we're within the last gradient_accumulation_steps samples)
                    # This prevents duplicate validation when global_step aligns with epoch end
                    samples_remaining = len(train_loader) - step - 1
                    is_near_epoch_end = samples_remaining < self.config.gradient_accumulation_steps
                    if self.global_step % self.config.eval_steps == 0 and not is_near_epoch_end and not getattr(self.config, 'skip_validation', False):
                        val_reward = self.validate(val_dataset)
                        self.val_rewards.append(val_reward)
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch validation
            if not getattr(self.config, 'skip_validation', False):
                val_reward = self.validate(val_dataset)
                self.val_rewards.append(val_reward)
                logger.info(f"Epoch {epoch + 1} - Avg Loss: {epoch_loss / len(train_loader):.4f}, Val Reward: {val_reward:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1} - Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save final model
        self.save_checkpoint("final")
        self.save_training_logs()
        
        logger.info("✅ Training complete!")
    
    def validate(self, val_dataset, num_samples: int = 50) -> float:
        """
        Validate model using evaluator.
        
        Args:
            val_dataset: Validation dataset
            num_samples: Number of samples to evaluate
            
        Returns:
            Average reward
        """
        result = self.evaluator.evaluate(
            dataset=val_dataset,
            num_samples=num_samples,
            detailed=False,
            description=f"Validation (step {self.global_step})"
        )
        
        # Save validation samples
        self.save_valid_samples(result.samples)
        
        # Save best model
        if result.avg_reward > self.best_val_reward:
            self.best_val_reward = result.avg_reward
            self.save_checkpoint("best")
            logger.info(f"  🎉 New best model! Reward: {result.avg_reward:.4f}")
        
        return result.avg_reward
    
    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'best', 'final', 'step_1000')
        """
        ckpt_dir = self.output_dir / 'ckpt_sft' / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        if self.config.use_lora:
            self.model.save_pretrained(ckpt_dir)
        else:
            self.model.save_pretrained(ckpt_dir)
        
        self.tokenizer.save_pretrained(ckpt_dir)
        
        logger.info(f"💾 Saved checkpoint: {ckpt_dir}")
    
    def save_valid_samples(self, samples: List[Dict]):
        """
        Save validation samples.
        
        Args:
            samples: List of validation sample results
        """
        output_file = self.output_dir / 'valid_samples' / f'step_{self.global_step}.json'
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
    
    def save_training_logs(self):
        """Save training logs and metrics."""
        logs = {
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_rewards': self.val_rewards,
            'best_val_reward': self.best_val_reward,
            'total_steps': self.global_step
        }
        
        log_file = self.output_dir / 'logs' / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"📝 Saved training logs: {log_file}")
    
    def save_manifest(self):
        """Save module manifest."""
        manifest = {
            'module': '03_sft_train',
            'version': '2.0.0',  # Updated version for refactored code
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'best_val_reward': self.best_val_reward,
            'total_steps': self.global_step
        }
        
        manifest_file = self.output_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Saved manifest: {manifest_file}")
