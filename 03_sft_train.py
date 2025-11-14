#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for FinQA
Trains a base LLM to generate valid JSON responses for financial QA.

This module performs standard supervised learning to teach the model basic skills
before RL training. Uses cross-entropy loss for training and reward function for monitoring.

Usage:
    python 03_sft_train.py --data_dir datasets/finqa_processed --base_model meta-llama/Meta-Llama-3-8B-Instruct
    python 03_sft_train.py --data_dir datasets/finqa_processed --base_model meta-llama/Meta-Llama-3-8B-Instruct --use_lora
    python 03_sft_train.py --data_dir datasets/finqa_processed --epochs 3 --batch_size 8 --lr 2e-5
"""

import argparse
import json
import logging
import torch
import importlib.util
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install transformers peft torch tqdm pyyaml")
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
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    output_dir: str = "outputs/run_001/03_sft"
    reward_spec: str = "outputs/run_001/02_rewards/reward_spec.yaml"
    
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Create config from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            SFTConfig instance
        """
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        config = cls()
        
        # Model settings
        if 'model' in yaml_config:
            config.base_model = yaml_config['model'].get('name', config.base_model)
        
        # LoRA settings
        if 'lora' in yaml_config:
            lora_cfg = yaml_config['lora']
            config.use_lora = lora_cfg.get('use_lora', config.use_lora)
            config.lora_r = lora_cfg.get('r', config.lora_r)
            config.lora_alpha = lora_cfg.get('alpha', config.lora_alpha)
            config.lora_dropout = lora_cfg.get('dropout', config.lora_dropout)
            # Store target_modules for later use
            config.lora_target_modules = lora_cfg.get('target_modules', None)
        
        # Training settings
        if 'training' in yaml_config:
            train_cfg = yaml_config['training']
            config.epochs = train_cfg.get('epochs', config.epochs)
            config.batch_size = train_cfg.get('batch_size', config.batch_size)
            config.learning_rate = train_cfg.get('learning_rate', config.learning_rate)
            config.warmup_steps = train_cfg.get('warmup_steps', config.warmup_steps)
            config.max_length = train_cfg.get('max_length', config.max_length)
            config.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', config.gradient_accumulation_steps)
            config.fp16 = train_cfg.get('fp16', config.fp16)
        
        # Validation settings
        if 'validation' in yaml_config:
            val_cfg = yaml_config['validation']
            config.eval_steps = val_cfg.get('eval_steps', config.eval_steps)
            config.save_steps = val_cfg.get('save_steps', config.save_steps)
            config.logging_steps = val_cfg.get('logging_steps', config.logging_steps)
        
        # Generation settings
        if 'generation' in yaml_config:
            gen_cfg = yaml_config['generation']
            config.max_new_tokens = gen_cfg.get('max_new_tokens', config.max_new_tokens)
            config.temperature = gen_cfg.get('temperature', config.temperature)
            config.top_p = gen_cfg.get('top_p', config.top_p)
        
        # Path settings
        if 'paths' in yaml_config:
            paths_cfg = yaml_config['paths']
            config.data_dir = paths_cfg.get('data_dir', config.data_dir)
            config.output_dir = paths_cfg.get('output_dir', config.output_dir)
            config.reward_spec = paths_cfg.get('reward_spec', config.reward_spec)
        
        return config


class FinQADataset(Dataset):
    """Dataset class for FinQA preprocessed data."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load preprocessed data (JSONL format)
        logger.info(f"Loading data from {data_file}")
        self.examples = []
        with open(data_file, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Add special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: input_text -> target (answer + program as JSON)
        input_text = example['input_text']
        
        # Create target as JSON string
        target_json = {
            'answer': example['target_answer'],
            'program': example.get('target_program', [])
        }
        target_text = json.dumps(target_json)
        
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
            'example_id': example.get('id', ''),
            'question': example.get('question', ''),
            'ground_truth': example.get('target_answer', '')
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
        
        # Determine target modules
        if hasattr(config, 'lora_target_modules') and config.lora_target_modules:
            target_modules = config.lora_target_modules
            logger.info(f"Using specified LoRA targets: {target_modules}")
        else:
            # Auto-detect based on model name
            model_name_lower = config.base_model.lower()
            if 'llama' in model_name_lower or 'mistral' in model_name_lower:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                logger.info(f"Detected Llama-style model, using targets: {target_modules}")
            elif 'gpt' in model_name_lower:
                target_modules = ["c_attn", "c_proj"]
                logger.info(f"Detected GPT-style model, using targets: {target_modules}")
            else:
                target_modules = None  # Let PEFT auto-detect
                logger.info("Using PEFT auto-detection for target modules")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for FinQA")
    
    # Config file argument
    parser.add_argument("--config", type=str,
                       help="Path to YAML config file (overrides other arguments)")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
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
    parser.add_argument("--output_dir", type=str, default="outputs/run_001/03_sft",
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
    
    # Create config from YAML or command line args
    if args.config:
        logger.info(f"📄 Loading config from: {args.config}")
        config = SFTConfig.from_yaml(args.config)
        # Command line args can still override YAML config
        if args.quick_test:
            config.epochs = 1
        if args.skip_validation:
            config.skip_validation = True
    else:
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
    train_file = data_dir / 'train.jsonl'
    val_file = data_dir / 'val.jsonl'
    
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
