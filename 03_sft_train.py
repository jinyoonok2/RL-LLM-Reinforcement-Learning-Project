#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for FinQA
Trains a base LLM to generate valid JSON responses for financial QA.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from utils.common import setup_logging, load_yaml_config
from utils.trainer import SFTTrainer
from utils.rewards import FinQARewardCalculator

logger = setup_logging()


@dataclass
class SFTConfig:
    """Simplified SFT configuration - loads from YAML."""
    # Core settings
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_lora: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-6
    warmup_steps: int = 100
    max_length: int = 2048
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = None
    
    # Validation
    eval_steps: int = 1546
    save_steps: int = 1546
    logging_steps: int = 100
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Paths
    data_dir: str = "datasets/finqa_processed"
    output_dir: str = "outputs/run_001/03_sft"
    reward_spec: str = "outputs/run_001/02_rewards/reward_spec.yaml"
    
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
        
        # Generation
        gen_cfg = cfg_dict.get('generation', {})
        for key in ['max_new_tokens', 'temperature', 'top_p']:
            if key in gen_cfg:
                setattr(config, key, gen_cfg[key])
        
        # Paths
        paths_cfg = cfg_dict.get('paths', {})
        for key in ['data_dir', 'output_dir', 'reward_spec']:
            if key in paths_cfg:
                setattr(config, key, paths_cfg[key])
        
        return config


class FinQADataset(Dataset):
    """Simplified FinQA dataset with automatic prompt filtering."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure pad token exists (same logic as setup_model)
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load and filter data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            all_examples = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(all_examples)} examples, filtering...")
        self.examples = self._filter_examples(all_examples)
        logger.info(f"Kept {len(self.examples)}/{len(all_examples)} samples "
                   f"({len(self.examples)/len(all_examples)*100:.1f}%)")
    
    def _filter_examples(self, examples):
        """Filter out examples where prompt is too long."""
        valid = []
        for ex in examples:
            prompt = self._build_prompt(ex)
            tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors='pt')
            if tokens['input_ids'].shape[1] < self.max_length - 50:  # Need room for answer
                valid.append(ex)
        return valid
    
    def _build_prompt(self, example):
        """Build instruction prompt from example."""
        return (f"{example['input_text']}\n\n"
                f"Question: {example.get('question', '')}\n\n"
                f"Provide your answer in JSON format with 'answer' and 'program' fields:\n")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = self._build_prompt(ex)
        
        # Create target JSON
        target = json.dumps({
            'answer': ex['target_answer'],
            'program': ex.get('target_program', [])
        })
        
        # Tokenize full sequence
        full_text = f"{prompt}{target}{self.tokenizer.eos_token}"
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Mask prompt tokens
        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors='pt')
        prompt_len = prompt_tokens['input_ids'].shape[1]
        
        labels = encodings['input_ids'].clone()
        labels[0, :prompt_len] = -100  # Mask prompt
        labels[0, encodings['attention_mask'][0] == 0] = -100  # Mask padding
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'example_id': ex.get('id', ''),
            'question': ex.get('question', ''),
            'ground_truth': ex.get('target_answer', '')
        }


def collate_fn(batch):
    """Stack batch tensors."""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
        'example_id': [x['example_id'] for x in batch],
        'question': [x['question'] for x in batch],
        'ground_truth': [x['ground_truth'] for x in batch]
    }


def setup_model(config: SFTConfig):
    """Load model with optional LoRA."""
    logger.info(f"Loading model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Handle pad token for different architectures
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new pad_token: [PAD]")
    
    # Determine dtype
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    logger.info(f"Using {dtype} for training")
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=dtype)
    
    # Apply LoRA
    if config.use_lora:
        logger.info("Applying LoRA adapters")
        target_modules = config.lora_target_modules
        if target_modules is None:
            # Auto-detect based on model architecture
            model_type = model.config.model_type.lower()
            if 'gpt2' in model_type or 'dialogpt' in config.base_model.lower():
                target_modules = ["c_attn", "c_proj"]  # GPT-2 style
            else:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Llama style
            logger.info(f"Auto-detected LoRA targets for {model_type}: {target_modules}")
        else:
            logger.info(f"Using configured LoRA targets: {target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    model = model.to(config.device)
    logger.info(f"Model moved to {config.device}")
    
    return model, tokenizer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for FinQA")
    parser.add_argument("--config", type=str, default="configs/models/llama-3.2-1b.yaml",
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
    logger.info("🚀 FinQA Supervised Fine-Tuning")
    logger.info("="*70)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"LoRA: {config.use_lora}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.learning_rate}")
    logger.info("="*70)
    
    # Load datasets
    data_dir = Path(config.data_dir)
    train_file, val_file = data_dir / 'train.jsonl', data_dir / 'val.jsonl'
    
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Data not found in {data_dir}. Run 01_prepare_dataset.py first!")
        return 1
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Load datasets
    train_dataset = FinQADataset(train_file, tokenizer, config.max_length)
    val_dataset = FinQADataset(val_file, tokenizer, config.max_length)
    
    # Quick test mode
    if args.quick_test:
        logger.info("🧪 Quick test mode: 100 train, 20 val samples")
        train_dataset.examples = train_dataset.examples[:100]
        val_dataset.examples = val_dataset.examples[:20]
        config.epochs = 1
        config.eval_steps = 50
    
    logger.info(f"✅ Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Load reward function
    try:
        reward_fn = FinQARewardCalculator()
        logger.info("✅ Loaded reward function")
    except Exception as e:
        logger.warning(f"Could not load reward function: {e}")
        reward_fn = None
    
    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn,
        collate_fn=collate_fn
    )
    
    logger.info("🚀 Starting SFT training")
    trainer.train(train_dataset, val_dataset)
    trainer.save_manifest()
    
    logger.info("\n🎉 Training Complete!")
    logger.info(f"📁 Outputs: {config.output_dir}")
    logger.info(f"💡 Best reward: {trainer.best_val_reward:.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
