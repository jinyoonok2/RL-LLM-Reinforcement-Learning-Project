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
        truncated_count = 0
        
        for ex in examples:
            prompt = self._build_prompt(ex)
            tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors='pt')
            prompt_length = tokens['input_ids'].shape[1]
            
            # Strategy 1: Keep if fits comfortably (80% of max_length)
            if prompt_length < int(self.max_length * 0.8):
                valid.append(ex)
            # Strategy 2: Truncate long examples but keep them if they're reasonable
            elif prompt_length < int(self.max_length * 0.95):  # Up to 95% is ok
                # Truncate the input text but keep the example
                truncated = self._truncate_example(ex)
                if truncated:
                    valid.append(truncated)
                    truncated_count += 1
                    
        if truncated_count > 0:
            logger.info(f"Truncated {truncated_count} long examples to fit context")
        return valid
    
    def _compress_financial_text(self, text):
        """Compress financial text by keeping only essential numerical data."""
        import re
        
        lines = text.split('\n')
        compressed_lines = []
        
        # Patterns for important financial data
        important_patterns = [
            r'\$[\d,\.]+',                    # Dollar amounts
            r'\d+\.?\d*\s*%',                 # Percentages  
            r'\b20\d{2}\b',                   # Years (2000-2099)
            r'\bQ[1-4]\b',                    # Quarters
            r'\d+\.?\d*\s*(million|billion|thousand|M|B|K)', # Large numbers
            r'\b(revenue|profit|income|expense|cost|sales|net|total)\b', # Key terms
            r'\b(increase|decreased?|growth|decline|up|down)\b',         # Change indicators
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
                
            # Keep lines with important patterns
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in important_patterns):
                # Clean up the line (remove excessive spaces)
                cleaned_line = re.sub(r'\s+', ' ', line)
                compressed_lines.append(cleaned_line)
                
            # Stop if we have enough information
            if len(compressed_lines) >= 12:
                break
        
        return '\n'.join(compressed_lines)

    def _truncate_example(self, example):
        """Truncate long examples intelligently with financial data focus."""
        original_input = example['input_text']
        
        # First try intelligent compression
        compressed_input = self._compress_financial_text(original_input)
        
        # If still too long, use truncation
        if len(compressed_input) > 800:
            compressed_input = compressed_input[:400] + "\n[...]\n" + compressed_input[-300:]
        
        # Only proceed if we got meaningful compression
        if len(compressed_input) < len(original_input) * 0.8:  # At least 20% reduction
            truncated_ex = example.copy()
            truncated_ex['input_text'] = compressed_input
            return truncated_ex
        
        return None
    
    def _build_prompt(self, example):
        """Build instruction prompt with realistic FinQA few-shot examples."""
        # Realistic FinQA-style examples to teach proper mathematical reasoning
        examples_text = '''You are a financial analyst. Read financial data carefully and calculate precise answers in JSON format.

Example 1:
Context: 
Table:
| year | revenue (millions) | expenses (millions) |
| 2019 | 2457 | 1823 |
| 2020 | 2156 | 1654 |

Question: What was the percentage decrease in revenue from 2019 to 2020?
Answer: {"answer": "12.25", "program": "divide(subtract(2457, 2156), 2457) * 100"}

Example 2:  
Context:
The company's payment volume was $637 billion across 5.0 billion transactions in 2020.

Question: What is the average payment volume per transaction?
Answer: {"answer": "127.40", "program": "divide(637, 5.0)"}

Example 3:
Context:
Total oil reserves: 450 million barrels. Canadian operations: 144 million barrels.

Question: What percentage of total oil reserves comes from Canada?
Answer: {"answer": "32.0", "program": "divide(144, 450) * 100"}

Now solve this problem using the same approach:
'''
        
        return (f"{examples_text}"
                f"Context: {example['input_text']}\n"
                f"Question: {example.get('question', '')}\n"
                f"Answer: ")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = self._build_prompt(ex)
        
        # Create target JSON with proper formatting
        program = ex.get('target_program', '')
        if isinstance(program, list):
            program = program[0] if program else ''
        
        target = json.dumps({
            'answer': str(ex['target_answer']),
            'program': str(program)
        }, separators=(',', ':'))  # Compact JSON
        
        # Ensure target isn't too long
        if len(target) > 200:  # Reasonable limit
            target = json.dumps({"answer": str(ex['target_answer']), "program": ""}, separators=(',', ':'))
            
        # Clean text to avoid tokenization issues
        prompt = prompt.replace('\x00', '').replace('\ufffd', '')
        target = target.replace('\x00', '').replace('\ufffd', '')
        
        # Tokenize full sequence with safety checks
        full_text = f"{prompt}{target}{self.tokenizer.eos_token}"
        
        try:
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=False  # DialoGPT specific
            )
        except Exception as e:
            logger.warning(f"Tokenization failed for example {idx}: {e}")
            # Fallback with simpler text
            simple_text = f"Question: {ex.get('question', '')[:200]}\nAnswer: {ex['target_answer'][:100]}{self.tokenizer.eos_token}"
            encodings = self.tokenizer(
                simple_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=False
            )
        
        # Tokenize prompt separately for masking
        try:
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, 
                                         return_tensors='pt', add_special_tokens=False)
            prompt_len = min(prompt_tokens['input_ids'].shape[1], self.max_length)
        except:
            prompt_len = len(prompt.split()) // 2  # Rough estimate fallback
        
        labels = encodings['input_ids'].clone()
        labels[0, :prompt_len] = -100  # Mask prompt
        labels[0, encodings['attention_mask'][0] == 0] = -100  # Mask padding
        
        # Validate token IDs are within vocabulary bounds
        vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)  # DialoGPT default
        
        # Replace invalid tokens with pad token ID
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        input_ids = encodings['input_ids'][0]
        invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
        input_ids[invalid_mask] = pad_token_id
        
        # Same for labels, but preserve -100
        labels_flat = labels[0]
        invalid_mask_labels = (labels_flat >= vocab_size) & (labels_flat != -100)
        labels_flat[invalid_mask_labels] = pad_token_id
        
        encodings['input_ids'][0] = input_ids
        labels[0] = labels_flat
        
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
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new pad_token: [PAD]")
    
    # Special handling for DialoGPT
    if 'dialogpt' in config.base_model.lower():
        # Ensure consistent tokenizer settings
        tokenizer.padding_side = 'left'  # GPT-2 style
        logger.info("Applied DialoGPT-specific tokenizer settings")
    
    # Determine dtype
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    logger.info(f"Using {dtype} for training")
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=dtype)
    
    # Resize token embeddings if we added new tokens
    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
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
    parser.add_argument("--test_tokenization", action="store_true",
                       help="Test tokenization only, don't train")
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
    
    # Test tokenization if requested
    if args.test_tokenization:
        logger.info("🧪 Testing tokenization on first 5 samples...")
        for i in range(min(5, len(train_dataset))):
            try:
                sample = train_dataset[i]
                logger.info(f"Sample {i}: input_ids shape={sample['input_ids'].shape}, max_id={sample['input_ids'].max()}")
            except Exception as e:
                logger.error(f"Sample {i} failed: {e}")
        return 0
    
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
