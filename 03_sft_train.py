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
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"  # Fast 1B model for testing
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Training settings
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 5e-6  # Lowered from 2e-5 to prevent NaN loss
    warmup_steps: int = 100
    max_length: int = 2048  # Balance between data coverage (~85-90%) and training speed
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0  # Gradient clipping to prevent exploding gradients
    
    # Validation settings
    eval_steps: int = 1563  # ~1 epoch (6251 samples / 4 grad_accum)
    save_steps: int = 1563  # Save at end of each epoch
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
    fp16: bool = False
    bf16: bool = torch.cuda.is_available()
    
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
            config.max_grad_norm = train_cfg.get('max_grad_norm', config.max_grad_norm)
            config.fp16 = train_cfg.get('fp16', config.fp16)
            config.bf16 = train_cfg.get('bf16', config.bf16)
        
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
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load preprocessed data (JSONL format)
        logger.info(f"Loading data from {data_file}")
        logger.info(f"Using max_length: {max_length}")
        logger.info(f"🔍 Dataset initialized with padding masking fix (v2)")
        self.examples = []
        with open(data_file, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Add special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Filter out samples that will have no trainable tokens
        # (This happens when prompts are too long to fit any answer)
        logger.info("Filtering samples with prompts that are too long...")
        original_count = len(self.examples)
        valid_examples = []
        
        for idx, example in enumerate(self.examples):
            # Quick check: tokenize just the prompt to see if it fits
            input_text = example['input_text']
            question = example.get('question', '')
            prompt = f"{input_text}\n\nQuestion: {question}\n\nProvide your answer in JSON format with 'answer' and 'program' fields:\n"
            
            prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors='pt')
            prompt_len = prompt_tokens['input_ids'].shape[1]
            
            # Keep only if there's room for at least 50 tokens of answer
            if prompt_len < max_length - 50:
                valid_examples.append(example)
        
        self.examples = valid_examples
        filtered_count = original_count - len(valid_examples)
        logger.info(f"  Filtered out {filtered_count}/{original_count} samples ({filtered_count/original_count*100:.1f}%)")
        logger.info(f"  Kept {len(valid_examples)} valid samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: input_text -> target (answer + program as JSON)
        input_text = example['input_text']
        question = example.get('question', '')
        
        # Create target as JSON string
        target_json = {
            'answer': example['target_answer'],
            'program': example.get('target_program', [])
        }
        target_text = json.dumps(target_json)
        
        # Create proper instruction format
        prompt = f"{input_text}\n\nQuestion: {question}\n\nProvide your answer in JSON format with 'answer' and 'program' fields:\n"
        
        # Combine for causal LM training (add EOS token)
        full_text = f"{prompt}{target_text}{self.tokenizer.eos_token}"
        
        # Tokenize the full text with truncation and padding
        full_encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize JUST the prompt to find where it ends (may be truncated)
        # This tells us where to start unmasking for the answer
        prompt_only_encodings = self.tokenizer(
            prompt,
            truncation=True,  # Truncate to match what happened in full_text
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_len = prompt_only_encodings['input_ids'].shape[1]
        
        labels = full_encodings['input_ids'].clone()
        
        # Mask the prompt part, keep the answer/target part for loss calculation
        # All invalid samples have been filtered out during __init__, so we just mask normally
        labels[0, :prompt_len] = -100
        # ALSO mask padding tokens (where attention_mask is 0)
        labels[0, full_encodings['attention_mask'][0] == 0] = -100
        
        return {
            'input_ids': full_encodings['input_ids'].squeeze(),
            'attention_mask': full_encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'example_id': example.get('id', ''),
            'question': example.get('question', ''),
            'ground_truth': example.get('target_answer', '')
        }


def collate_fn(batch):
    """Custom collate function to properly batch samples without corrupting labels."""
    # Stack tensors manually to ensure labels aren't corrupted
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'example_id': [item['example_id'] for item in batch],
        'question': [item['question'] for item in batch],
        'ground_truth': [item['ground_truth'] for item in batch]
    }


def load_reward_function():
    """Load reward function from utils.rewards."""
    try:
        from utils.rewards import FinQARewardCalculator
        reward_fn = FinQARewardCalculator()
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
    
    # Determine the correct torch dtype
    dtype = torch.float32  # Default
    if config.bf16:
        dtype = torch.bfloat16
        logger.info("Using bfloat16 for training.")
    elif config.fp16:
        dtype = torch.float16
        logger.info("Using float16 for training.")
    else:
        logger.info("Using float32 for training.")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype
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
    
    # Move model to device
    model = model.to(config.device)
    logger.info(f"Model moved to {config.device}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for FinQA")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="configs/models/llama-3.2-1b.yaml",
                       help="Path to YAML config file (default: 1B model, use configs/models/llama-3-8b.yaml for 8B)")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (full fine-tuning - requires more VRAM)")
    parser.add_argument("--lora_r", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, default="datasets/finqa_processed",
                       help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="outputs/run_001/03_sft",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
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
    
    # Datasets loaded and filtered - ready to train
    logger.info(f"✅ Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")
    
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
        reward_fn=reward_fn,
        collate_fn=collate_fn  # Use custom collate to preserve labels
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
