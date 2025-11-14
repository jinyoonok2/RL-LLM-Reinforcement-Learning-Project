#!/usr/bin/env python3
"""
Model Architecture Inspector
Detects model architecture and generates model-specific config files.

This script:
1. Loads a model and inspects its layer names
2. Detects appropriate LoRA target modules
3. Generates YAML config files for training

Usage:
    python inspect_model_architecture.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python inspect_model_architecture.py --model meta-llama/Llama-3-8B-Instruct
    python inspect_model_architecture.py --all  # Generate configs for all default models
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install transformers torch pyyaml")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_model_layers(model_name: str) -> Dict:
    """
    Inspect model architecture and detect layer names.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Dict with model info and detected layers
    """
    logger.info(f"🔍 Inspecting model: {model_name}")
    
    try:
        # Load config only (lightweight)
        config = AutoConfig.from_pretrained(model_name)
        logger.info(f"  Model type: {config.model_type}")
        logger.info(f"  Architecture: {config.architectures}")
        
        # Load model architecture (we'll use small float32 to inspect structure)
        logger.info("  Loading model to inspect layers...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Keep on CPU for inspection
        )
        
        # Collect all layer names
        layer_names = set()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_type = type(module).__name__
                # Extract just the layer name pattern (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
                if '.' in name:
                    layer_name = name.split('.')[-1]
                    layer_names.add(layer_name)
        
        logger.info(f"  Found {len(layer_names)} unique layer types")
        
        # Detect LoRA target modules based on common patterns
        lora_targets = detect_lora_targets(layer_names, config.model_type)
        
        model_info = {
            'model_name': model_name,
            'model_type': config.model_type,
            'architecture': config.architectures[0] if config.architectures else 'Unknown',
            'hidden_size': getattr(config, 'hidden_size', None),
            'num_layers': getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', None),
            'vocab_size': config.vocab_size,
            'all_layer_names': sorted(layer_names),
            'recommended_lora_targets': lora_targets
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to inspect model: {e}")
        return None


def detect_lora_targets(layer_names: set, model_type: str) -> List[str]:
    """
    Detect appropriate LoRA target modules based on layer names.
    
    Args:
        layer_names: Set of all layer names in the model
        model_type: Model type from config
        
    Returns:
        List of recommended LoRA target module names
    """
    # Common attention projection layers
    attention_patterns = {
        'llama': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'gpt2': ['c_attn', 'c_proj'],
        'gpt_neox': ['query_key_value', 'dense'],
        'opt': ['q_proj', 'v_proj', 'k_proj', 'out_proj'],
        'bloom': ['query_key_value', 'dense'],
    }
    
    # Try to match model type
    targets = []
    for pattern_type, pattern_layers in attention_patterns.items():
        if pattern_type in model_type.lower():
            # Verify these layers actually exist
            targets = [layer for layer in pattern_layers if layer in layer_names]
            if targets:
                logger.info(f"  Detected {pattern_type} architecture")
                logger.info(f"  Recommended LoRA targets: {targets}")
                return targets
    
    # Fallback: look for common attention projection patterns
    common_projections = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'c_attn', 'c_proj', 
                         'query_key_value', 'dense', 'qkv_proj', 'out_proj']
    
    targets = [layer for layer in common_projections if layer in layer_names]
    
    if targets:
        logger.info(f"  Using detected attention layers: {targets}")
    else:
        logger.warning(f"  Could not detect attention layers! Available: {sorted(layer_names)}")
        targets = None  # Let PEFT auto-detect
    
    return targets


def generate_model_config(model_info: Dict, output_dir: Path) -> Path:
    """
    Generate training config YAML for a specific model.
    
    Args:
        model_info: Model information from inspection
        output_dir: Directory to save config
        
    Returns:
        Path to generated config file
    """
    # Create safe filename from model name
    safe_name = model_info['model_name'].replace('/', '_').replace('-', '_')
    config_file = output_dir / f"config_{safe_name}.yaml"
    
    # Determine appropriate settings based on model size
    hidden_size = model_info.get('hidden_size', 768)
    
    # Smaller models (< 1B params): can use larger batch size
    # Larger models (> 1B params): need smaller batch size
    if hidden_size < 1024:
        batch_size = 8
        gradient_accum = 2
        lora_r = 8
    elif hidden_size < 2048:
        batch_size = 4
        gradient_accum = 4
        lora_r = 16
    else:
        batch_size = 2
        gradient_accum = 8
        lora_r = 32
    
    config = {
        'model': {
            'name': model_info['model_name'],
            'type': model_info['model_type'],
            'architecture': model_info['architecture'],
            'hidden_size': hidden_size,
            'num_layers': model_info.get('num_layers'),
            'vocab_size': model_info.get('vocab_size'),
        },
        'lora': {
            'use_lora': True,
            'r': lora_r,
            'alpha': lora_r * 2,
            'dropout': 0.05,
            'target_modules': model_info['recommended_lora_targets'],
        },
        'training': {
            'epochs': 3,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accum,
            'learning_rate': 2e-5,
            'warmup_steps': 100,
            'max_length': 512,
            'fp16': True,
        },
        'validation': {
            'eval_steps': 500,
            'save_steps': 1000,
            'logging_steps': 100,
        },
        'generation': {
            'max_new_tokens': 128,
            'temperature': 0.7,
            'top_p': 0.9,
        },
        'paths': {
            'data_dir': 'datasets/finqa_processed',
            'output_dir': f'outputs/run_001/03_sft_{safe_name}',
            'reward_spec': 'outputs/run_001/02_rewards/reward_spec.yaml',
        },
        'metadata': {
            'generated_by': 'inspect_model_architecture.py',
            'all_available_layers': model_info['all_layer_names'],
        }
    }
    
    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✅ Generated config: {config_file}")
    return config_file


def print_model_summary(model_info: Dict):
    """Print a summary of the model inspection."""
    print("\n" + "="*70)
    print(f"📋 Model Summary: {model_info['model_name']}")
    print("="*70)
    print(f"Type:         {model_info['model_type']}")
    print(f"Architecture: {model_info['architecture']}")
    print(f"Hidden Size:  {model_info['hidden_size']}")
    print(f"Num Layers:   {model_info['num_layers']}")
    print(f"Vocab Size:   {model_info['vocab_size']}")
    print(f"\n🎯 Recommended LoRA Targets:")
    if model_info['recommended_lora_targets']:
        for target in model_info['recommended_lora_targets']:
            print(f"  - {target}")
    else:
        print(f"  - Auto-detect (will use PEFT defaults)")
    print(f"\n📦 All Available Layer Types ({len(model_info['all_layer_names'])}):")
    for i, layer in enumerate(sorted(model_info['all_layer_names']), 1):
        print(f"  {i:2d}. {layer}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect model architecture and generate configs")
    parser.add_argument("--model", type=str, 
                       help="Model name to inspect (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--all", action="store_true",
                       help="Generate configs for all default models")
    parser.add_argument("--output_dir", type=str, default="configs/models",
                       help="Directory to save config files")
    parser.add_argument("--no_config", action="store_true",
                       help="Only inspect, don't generate config files")
    
    args = parser.parse_args()
    
    # Default models to inspect if --all is specified
    default_models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/DialoGPT-medium",
    ]
    
    if args.all:
        models_to_inspect = default_models
    elif args.model:
        models_to_inspect = [args.model]
    else:
        parser.print_help()
        print("\n❌ Error: Specify --model <name> or --all")
        return 1
    
    output_dir = Path(args.output_dir)
    
    logger.info("🚀 Starting Model Architecture Inspection")
    logger.info("="*70)
    
    results = {}
    for model_name in models_to_inspect:
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        model_info = inspect_model_layers(model_name)
        
        if model_info:
            results[model_name] = model_info
            print_model_summary(model_info)
            
            if not args.no_config:
                config_file = generate_model_config(model_info, output_dir)
                print(f"💾 Config saved: {config_file}\n")
        else:
            logger.error(f"❌ Failed to inspect {model_name}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Inspection Summary")
    print("="*70)
    print(f"Models inspected: {len(results)}/{len(models_to_inspect)}")
    
    if results and not args.no_config:
        print(f"\n✅ Generated {len(results)} config files in: {output_dir}")
        print("\n💡 Next steps:")
        print("  1. Review the generated YAML configs")
        print("  2. Adjust batch_size/learning_rate if needed")
        print("  3. Use config in training:")
        for model_name in results.keys():
            safe_name = model_name.replace('/', '_').replace('-', '_')
            print(f"     python 03_sft_train.py --config configs/models/config_{safe_name}.yaml")
    
    return 0


if __name__ == "__main__":
    exit(main())
