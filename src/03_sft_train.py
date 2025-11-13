#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for FinQA
Trains a base LLM to generate valid JSON responses for financial QA.

This module performs standard supervised learning to teach the model basic skills
before RL training. Uses cross-entropy loss for training and reward function for monitoring.

Usage:
    python src/03_sft_train.py --data_dir datasets/finqa_processed --base_model microsoft/DialoGPT-medium
    python src/03_sft_train.py --data_dir datasets/finqa_processed --base_model microsoft/DialoGPT-medium --use_lora
    python src/03_sft_train.py --data_dir datasets/finqa_processed --epochs 3 --batch_size 8 --lr 2e-5
"""

import argparse
import json
import logging
import torch
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import numpy as np

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup
    )
    from torch.utils.data import Dataset, DataLoader
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install transformers peft torch tqdm")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model settings
    base_model: str = "microsoft/DialoGPT-medium"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Training settings
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Validation settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Generation settings for validation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Paths
    data_dir: str = "datasets/finqa_processed"
    output_dir: str = "outputs/finqa_rl/run_001/03_sft"
    reward_spec: str = "outputs/finqa_rl/02_rewards/reward_spec.yaml"
    
    # Other
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config


class FinQADataset(Dataset):
    """Dataset class for FinQA preprocessed data."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load preprocessed data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.examples = data['examples']
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Add special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: input_text -> target_text
        input_text = example['input_text']
        target_text = example['target_text']
        
        # Combine for causal LM training
        full_text = f"{input_text} {target_text}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (mask input part, only compute loss on target)
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_len = input_encodings['input_ids'].shape[1]
        labels = encodings['input_ids'].clone()
        labels[0, :input_len] = -100  # Ignore input tokens in loss
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'example_id': example['id'],
            'question': example['question'],
            'ground_truth': example['ground_truth_answer']
        }


class SFTTrainer:
    """Trainer for supervised fine-tuning."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'ckpt_sft').mkdir(exist_ok=True)
        (self.output_dir / 'valid_samples').mkdir(exist_ok=True)
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        )
        
        # Apply LoRA if specified
        if config.use_lora:
            logger.info("Applying LoRA adapters")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["c_attn", "c_proj"]  # For GPT-2 style models
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        self.model.to(self.device)
        
        # Load reward function for validation
        self.reward_fn = self._load_reward_function()
        
        # Training tracking
        self.global_step = 0
        self.best_val_reward = -float('inf')
        self.train_losses = []
        self.val_rewards = []
    
    def _load_reward_function(self):
        """Load reward function from 02_build_rewards.py."""
        try:
            spec = importlib.util.spec_from_file_location(
                "rewards",
                Path(__file__).parent / "02_build_rewards.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            reward_fn = module.FinQARewardFunction()
            logger.info("✅ Loaded reward function for validation")
            return reward_fn
        except Exception as e:
            logger.warning(f"Could not load reward function: {e}")
            return None
    
    def train(self, train_dataset: FinQADataset, val_dataset: FinQADataset):
        """Main training loop."""
        logger.info("🚀 Starting SFT training")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
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
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
                    
                    # Validation
                    if self.global_step % self.config.eval_steps == 0 and not getattr(self.config, 'skip_validation', False):
                        val_reward = self.validate(val_dataset)
                        self.val_rewards.append(val_reward)
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch validation
            if not getattr(self.config, 'skip_validation', False):
                val_reward = self.validate(val_dataset)
                logger.info(f"Epoch {epoch + 1} - Avg Loss: {epoch_loss / len(train_loader):.4f}, Val Reward: {val_reward:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1} - Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save final model
        self.save_checkpoint("final")
        self.save_training_logs()
        
        logger.info("✅ Training complete!")
    
    def validate(self, val_dataset: FinQADataset, num_samples: int = 50) -> float:
        """Validate model and compute average reward."""
        logger.info("📊 Running validation...")
        self.model.eval()
        
        total_reward = 0
        valid_samples = []
        parse_success = 0
        
        # Sample random examples
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Validation"):
                example = val_dataset[idx]
                
                # Generate prediction
                input_text = val_dataset.examples[idx]['input_text']
                prediction = self.generate(input_text)
                ground_truth = example['ground_truth']
                
                # Calculate reward if reward function available
                if self.reward_fn:
                    reward = self.reward_fn.calculate(
                        prediction=prediction,
                        ground_truth=ground_truth,
                        question=example['question']
                    )
                    total_reward += reward.total
                    
                    # Check if valid JSON
                    try:
                        json.loads(prediction)
                        parse_success += 1
                        parse_ok = True
                    except:
                        parse_ok = False
                    
                    valid_samples.append({
                        'example_id': example['example_id'],
                        'question': example['question'][:100],
                        'prediction': prediction[:200],
                        'ground_truth': ground_truth,
                        'reward': reward.total,
                        'parse_ok': parse_ok
                    })
        
        avg_reward = total_reward / len(indices) if len(indices) > 0 else 0
        parse_rate = parse_success / len(indices) if len(indices) > 0 else 0
        
        # Save validation samples
        self.save_valid_samples(valid_samples)
        
        logger.info(f"Validation Results:")
        logger.info(f"  Avg Reward: {avg_reward:.4f}")
        logger.info(f"  Parse Rate: {parse_rate:.2%}")
        
        # Save best model
        if avg_reward > self.best_val_reward:
            self.best_val_reward = avg_reward
            self.save_checkpoint("best")
            logger.info(f"  🎉 New best model! Reward: {avg_reward:.4f}")
        
        return avg_reward
    
    def generate(self, input_text: str) -> str:
        """Generate prediction for input text with robust error handling."""
        try:
            # Move to CPU to avoid CUDA issues during generation
            self.model.cpu()
            
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length
            )
            
            # Use greedy decoding (most stable)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=min(self.config.max_new_tokens, 64),  # Limit length
                    do_sample=False,  # Greedy (no sampling)
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Move back to device
            self.model.to(self.device)
            
            # Decode only the generated part
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return prediction.strip()
        
        except Exception as e:
            # Move back to device even on error
            self.model.to(self.device)
            logger.warning(f"Generation failed: {str(e)[:100]}, returning empty")
            return ""
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
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
        """Save validation samples."""
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
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'best_val_reward': self.best_val_reward,
            'total_steps': self.global_step
        }
        
        manifest_file = self.output_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Saved manifest: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for FinQA")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium",
                       help="Base model to fine-tune")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, default="datasets/finqa_processed",
                       help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="outputs/finqa_rl/run_001/03_sft",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Other arguments
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with small subset of data")
    parser.add_argument("--skip_validation", action="store_true",
                       help="Skip validation during training (faster, for testing)")
    
    args = parser.parse_args()
    
    # Create config
    config = SFTConfig.from_args(args)
    
    logger.info("="*70)
    logger.info("🚀 FinQA Supervised Fine-Tuning")
    logger.info("="*70)
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"Use LoRA: {config.use_lora}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info("="*70)
    
    # Load datasets
    data_dir = Path(config.data_dir)
    train_file = data_dir / 'train_processed.json'
    val_file = data_dir / 'val_processed.json'
    
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Preprocessed data not found in {data_dir}")
        logger.error("Run 01_prepare_dataset.py first!")
        return 1
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    # Load tokenizer for datasets
    train_dataset = FinQADataset(train_file, trainer.tokenizer, config.max_length)
    val_dataset = FinQADataset(val_file, trainer.tokenizer, config.max_length)
    
    # Quick test mode
    if args.quick_test:
        logger.info("🧪 Running in quick test mode (100 train, 20 val examples)")
        train_dataset.examples = train_dataset.examples[:100]
        val_dataset.examples = val_dataset.examples[:20]
        config.epochs = 1
        config.eval_steps = 50
        config.save_steps = 100
    
    # Skip validation if requested
    if args.skip_validation:
        config.skip_validation = True
        logger.info("⏭️  Skipping validation (training only)")
    
    # Train
    trainer.train(train_dataset, val_dataset)
    
    # Save manifest
    trainer.save_manifest()
    
    logger.info("\n🎉 SFT Training Complete!")
    logger.info(f"📁 Outputs saved to: {config.output_dir}")
    logger.info(f"💡 Best validation reward: {trainer.best_val_reward:.4f}")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"  1. Check validation samples: cat {config.output_dir}/valid_samples/step_*.json")
    logger.info(f"  2. Review training logs: cat {config.output_dir}/logs/training_log.json")
    logger.info(f"  3. Use checkpoint for RL: {config.output_dir}/ckpt_sft/best/")
    
    return 0


if __name__ == "__main__":
    exit(main())
