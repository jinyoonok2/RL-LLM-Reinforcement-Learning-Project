#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for FinQA
Trains a base LLM to generate valid JSON responses for financial QA.

This module performs standard supervised learning to teach the model basic skills
before RL training. Uses cross-entropy loss for training and reward function for monitoring.

Usage:
    python 03_sft_train.py --data_dir datasets/finqa_processed --base_model microsoft/DialoGPT-medium
    python 03_sft_train.py --data_dir datasets/finqa_processed --base_model microsoft/DialoGPT-medium --use_lora
    python 03_sft_train.py --data_dir datasets/finqa_processed --epochs 3 --batch_size 8 --lr 2e-5
"""

import argparse
import json
import logging
import torch
import importlib.util
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install transformers peft torch tqdm")
    exit(1)

# Import shared utilities
from utils.trainer import SFTTrainer

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


def load_reward_function():
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


def setup_model(config: SFTConfig):
    """
    Setup model and tokenizer with optional LoRA.
    
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
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
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


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
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("="*70)
    logger.info("🚀 FinQA Supervised Fine-Tuning (Refactored)")
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
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(config)
    
    # Load datasets
    train_dataset = FinQADataset(train_file, tokenizer, config.max_length)
    val_dataset = FinQADataset(val_file, tokenizer, config.max_length)
    
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
    
    # Load reward function
    reward_fn = load_reward_function()
    
    # Create trainer with utilities
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn
    )
    
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
