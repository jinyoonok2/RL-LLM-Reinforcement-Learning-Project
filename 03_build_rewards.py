#!/usr/bin/env python3
"""
FinQA Reward Calculator for Candidates - Calculate rewards for all candidate answers.

Usage:
    python 03_build_rewards.py --input_dir datasets/finqa_candidates --output_dir datasets/finqa_with_rewards
    python 03_build_rewards.py --run_tests
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from utils.common import setup_logging, save_manifest, save_json_data, print_section
from utils.rewards import FinQARewardCalculator, RewardComponents

logger = setup_logging()


def calculate_rewards_for_candidates(input_dir: str, output_dir: str, weights: dict = None):
    """
    Calculate rewards for all candidates in the dataset.
    
    Args:
        input_dir: Input directory with candidate datasets (from 02_generate_candidates.py)
        output_dir: Output directory for datasets with rewards
        weights: Component weights (if None, uses defaults)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reward_calc = FinQARewardCalculator(weights=weights)
    
    splits = ['train', 'val', 'test']
    stats = {
        'total_questions': 0,
        'total_candidates': 0,
        'by_split': {},
        'reward_stats': {
            'mean': 0.0,
            'max': 0.0,
            'min': float('inf'),
            'gold_rewards': [],
            'corrupted_rewards': []
        }
    }
    
    logger.info(f"Calculating rewards for candidates from {input_dir}")
    
    for split in splits:
        input_file = input_dir / f"{split}.jsonl"
        if not input_file.exists():
            logger.warning(f"Skipping {split}: file not found")
            continue
        
        logger.info(f"Processing {split} split...")
        
        # Load candidate dataset
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Calculate rewards for each candidate
        output_examples = []
        all_rewards = []
        gold_rewards = []
        corrupted_rewards = []
        
        for ex in tqdm(examples, desc=f"Calculating rewards ({split})"):
            candidates_with_rewards = []
            
            # Get gold answer from first candidate (is_gold=True)
            gold_candidate = next((c for c in ex['candidates'] if c['is_gold']), None)
            if not gold_candidate:
                logger.warning(f"No gold candidate found for question {ex['id']}")
                gold_answer = ""
            else:
                gold_answer = gold_candidate['answer']
            
            # Calculate reward for each candidate
            for cand in ex['candidates']:
                prediction = cand['answer']
                predicted_program = cand.get('program', [])
                
                # Calculate component rewards
                components = reward_calc.calculate_component_rewards(
                    prediction=prediction,
                    ground_truth=gold_answer,
                    predicted_program=predicted_program if isinstance(predicted_program, list) else None,
                    ground_truth_program=gold_candidate.get('program', []) if gold_candidate else None
                )
                
                # Calculate total reward (use the total field from components)
                total_reward = components.total
                
                # Add reward info to candidate
                cand_with_reward = {
                    **cand,
                    'reward': total_reward,
                    'reward_components': {
                        'exact_match': components.exact_match,
                        'numerical_close': components.numerical_close,
                        'program_valid': components.program_valid,
                        'format_valid': components.format_valid,
                        'total': components.total
                    }
                }
                candidates_with_rewards.append(cand_with_reward)
                
                # Track statistics
                all_rewards.append(total_reward)
                if cand['is_gold']:
                    gold_rewards.append(total_reward)
                else:
                    corrupted_rewards.append(total_reward)
            
            output_examples.append({
                'id': ex['id'],
                'input_text': ex['input_text'],
                'question': ex['question'],
                'candidates': candidates_with_rewards,
                'metadata': ex.get('metadata', {})
            })
        
        # Save output
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in output_examples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(output_examples)} questions with rewards")
        
        # Calculate split statistics
        split_stats = {
            'questions': len(output_examples),
            'candidates': len(all_rewards),
            'reward_stats': {
                'mean': sum(all_rewards) / len(all_rewards) if all_rewards else 0,
                'max': max(all_rewards) if all_rewards else 0,
                'min': min(all_rewards) if all_rewards else 0,
                'gold_mean': sum(gold_rewards) / len(gold_rewards) if gold_rewards else 0,
                'corrupted_mean': sum(corrupted_rewards) / len(corrupted_rewards) if corrupted_rewards else 0
            }
        }
        
        stats['by_split'][split] = split_stats
        stats['total_questions'] += len(output_examples)
        stats['total_candidates'] += len(all_rewards)
        stats['reward_stats']['gold_rewards'].extend(gold_rewards)
        stats['reward_stats']['corrupted_rewards'].extend(corrupted_rewards)
    
    # Calculate overall statistics
    all_gold = stats['reward_stats']['gold_rewards']
    all_corrupted = stats['reward_stats']['corrupted_rewards']
    all_combined = all_gold + all_corrupted
    
    stats['reward_stats']['mean'] = sum(all_combined) / len(all_combined) if all_combined else 0
    stats['reward_stats']['max'] = max(all_combined) if all_combined else 0
    stats['reward_stats']['min'] = min(all_combined) if all_combined else 0
    stats['reward_stats']['gold_mean'] = sum(all_gold) / len(all_gold) if all_gold else 0
    stats['reward_stats']['corrupted_mean'] = sum(all_corrupted) / len(all_corrupted) if all_corrupted else 0
    
    # Clean up large lists for saving
    stats['reward_stats'].pop('gold_rewards')
    stats['reward_stats'].pop('corrupted_rewards')
    
    # Save statistics
    stats_file = output_dir / 'reward_stats.json'
    save_json_data(stats, str(stats_file))
    
    save_manifest(
        output_dir,
        '03_build_rewards',
        {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'weights': weights or 'default'
        },
        stats
    )
    
    logger.info(f"✅ Reward calculation complete")
    logger.info(f"   Questions: {stats['total_questions']}")
    logger.info(f"   Total candidates: {stats['total_candidates']}")
    logger.info(f"   Average reward: {stats['reward_stats']['mean']:.3f}")
    logger.info(f"   Gold avg: {stats['reward_stats']['gold_mean']:.3f}")
    logger.info(f"   Corrupted avg: {stats['reward_stats']['corrupted_mean']:.3f}")


def parse_weights(weights_str: str) -> dict:
    """Parse weights string like 'exact=1.0,numerical=0.8'."""
    weights = {}
    for item in weights_str.split(','):
        key, value = item.strip().split('=')
        weights[key.strip()] = float(value.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(description="Calculate rewards for candidate answers")
    parser.add_argument("--input_dir", type=str, default="datasets/finqa_candidates",
                       help="Input directory with candidate datasets")
    parser.add_argument("--output_dir", type=str, default="datasets/finqa_with_rewards",
                       help="Output directory for datasets with rewards")
    parser.add_argument("--weights", type=str,
                       help="Component weights (comma-separated key=value pairs)")
    parser.add_argument("--run_tests", action="store_true",
                       help="Run unit tests on reward function before processing")
    args = parser.parse_args()
    
    # Parse weights if provided
    weights = parse_weights(args.weights) if args.weights else None
    
    # Run tests if requested
    if args.run_tests:
        reward_calc = FinQARewardCalculator(weights=weights)
        test_passed = run_reward_tests(reward_calc, weights or {})
        if not test_passed:
            logger.warning("Some tests failed, but continuing...")
    
    calculate_rewards_for_candidates(args.input_dir, args.output_dir, weights)


if __name__ == "__main__":
    main()
