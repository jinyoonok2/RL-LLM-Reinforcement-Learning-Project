#!/usr/bin/env python3
"""
FinQA Dataset Preprocessing - Prepares data for RL training with reward labels.

Usage:
    python 01_prepare_dataset.py --input_dir datasets/finqa --output_dir datasets/finqa_processed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from utils.common import setup_logging, save_manifest, save_json_data, load_json_data
from utils.rewards import FinQARewardCalculator

logger = setup_logging()


def format_context(example: Dict) -> str:
    """Format question context from FinQA example."""
    parts = []
    
    # Handle pre_text (can be string or list)
    if example.get('pre_text'):
        pre_text = example['pre_text']
        if isinstance(pre_text, list):
            parts.extend(pre_text)
        else:
            parts.append(pre_text)
    
    # Handle table
    if example.get('table'):
        table = example['table']
        if isinstance(table, list) and table:
            parts.append("\nTable:")
            for row in table:
                if isinstance(row, list):
                    parts.append("  | " + " | ".join(str(cell) for cell in row))
    
    # Handle post_text (can be string or list)
    if example.get('post_text'):
        post_text = example['post_text']
        if isinstance(post_text, list):
            parts.extend(post_text)
        else:
            parts.append(post_text)
    
    return "\n".join(parts)


def prepare_finqa_dataset(input_dir: str, output_dir: str, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Prepare FinQA dataset for training.
    
    Args:
        input_dir: Path to raw FinQA dataset
        output_dir: Path to save processed dataset
        model_name: Model name for tokenizer (unused currently, reserved for future)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reward_calc = FinQARewardCalculator()
    splits = ['train', 'val', 'test']
    stats = {'total': 0, 'by_split': {}, 'reward_distribution': {'exact': 0, 'partial': 0, 'wrong': 0}}
    
    logger.info(f"Processing FinQA dataset from {input_dir}")
    
    for split in splits:
        input_file = input_dir / f"{split}.json"
        if not input_file.exists():
            logger.warning(f"Skipping {split}: file not found")
            continue
        
        logger.info(f"Processing {split} split...")
        raw_data = load_json_data(str(input_file))
        processed = []
        
        for ex in raw_data:
            qa = ex.get('qa', {})
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            program = qa.get('program', [])
            
            # Format input
            context = format_context(ex)
            input_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            
            # Calculate ternary reward for this example
            reward_value, explanation = reward_calc.calculate_ternary_reward(answer, answer, question)
            
            # Map ternary reward to category
            if reward_value == 1:
                reward_type = 'exact'
                stats['reward_distribution']['exact'] += 1
            elif reward_value == 0:
                reward_type = 'partial'
                stats['reward_distribution']['partial'] += 1
            else:
                reward_type = 'wrong'
                stats['reward_distribution']['wrong'] += 1
            
            processed.append({
                'id': ex.get('id', f"{split}_{len(processed)}"),
                'input_text': input_text,
                'target_answer': str(answer),
                'target_program': program,
                'question': question,
                'reward_type': reward_type,
                'metadata': {
                    'split': split,
                    'has_program': len(program) > 0 if isinstance(program, list) else False
                }
            })
        
        # Save processed split
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(processed)} examples to {output_file}")
        stats['by_split'][split] = len(processed)
        stats['total'] += len(processed)
    
    # Save statistics and manifest
    stats_file = output_dir / 'preprocessing_stats.json'
    save_json_data(stats, str(stats_file))
    
    save_manifest(
        output_dir,
        '01_prepare_dataset',
        {'input_dir': str(input_dir), 'output_dir': str(output_dir), 'model_name': model_name},
        stats
    )
    
    logger.info(f"✅ Preprocessing complete: {stats['total']} total examples")
    logger.info(f"   Reward distribution: {stats['reward_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FinQA dataset for training")
    parser.add_argument("--input_dir", type=str, default="datasets/finqa",
                       help="Input directory with raw FinQA data")
    parser.add_argument("--output_dir", type=str, default="datasets/finqa_processed",
                       help="Output directory for processed data")
    parser.add_argument("--config", type=str,
                       help="Path to model config YAML file (e.g., configs/models/config_meta_llama_Llama_3_8B_Instruct.yaml)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model name for tokenizer (overridden by --config if provided)")
    args = parser.parse_args()
    
    # Load model name from config if provided
    model_name = args.model_name
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model', {}).get('name', model_name)
        print(f"Using model from config: {model_name}")
    
    prepare_finqa_dataset(args.input_dir, args.output_dir, model_name)


if __name__ == "__main__":
    main()
