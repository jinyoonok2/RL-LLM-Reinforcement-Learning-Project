#!/usr/bin/env python3
"""
Generate K Candidates per Prompt for RL Training
Samples multiple outputs from SFT model to use in PPO/GRPO/RLOO training.

Usage:
    python 04_generate_candidates.py --policy_ckpt outputs/run_001/03_sft
    python 04_generate_candidates.py --policy_ckpt outputs/run_001/03_sft --num_candidates 8 --split val
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import setup_logging, save_manifest, load_json_data, save_json_data
from utils.rewards import FinQARewardCalculator

logger = setup_logging()


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda"
) -> List[str]:
    """Generate K candidate responses for a single prompt."""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    candidates = []
    with torch.no_grad():
        for _ in range(num_candidates):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            
            # Decode only the generated part (skip the prompt)
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            candidates.append(generated_text.strip())
    
    return candidates


def parse_json_response(response: str) -> tuple[Dict, bool]:
    """Try to parse JSON from response text."""
    try:
        # Try direct parsing
        parsed = json.loads(response)
        return parsed, True
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                try:
                    parsed = json.loads(response[start:end].strip())
                    return parsed, True
                except json.JSONDecodeError:
                    pass
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                try:
                    parsed = json.loads(response[start:end].strip())
                    return parsed, True
                except json.JSONDecodeError:
                    pass
        
        return {}, False


def process_dataset(
    data_file: Path,
    policy_ckpt: Path,
    output_dir: Path,
    num_candidates: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_samples: int = None,
    device: str = "cuda"
):
    """Generate candidates for all examples in dataset."""
    
    logger.info(f"Loading model from {policy_ckpt}")
    model = AutoModelForCausalLM.from_pretrained(policy_ckpt).to(device)
    tokenizer = AutoTokenizer.from_pretrained(policy_ckpt)
    model.eval()
    
    # Load data
    logger.info(f"Loading data from {data_file}")
    examples = load_json_data(data_file)
    if max_samples:
        examples = examples[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Initialize reward calculator for scoring
    reward_calc = FinQARewardCalculator()
    
    # Generate candidates
    logger.info(f"Generating {num_candidates} candidates per prompt...")
    candidates_data = []
    scores_data = []
    
    parse_success = 0
    total_candidates = 0
    
    for example in tqdm(examples, desc="Generating candidates"):
        prompt = example['prompt']
        target = example.get('target_json', {})
        
        # Generate K candidates
        candidate_texts = generate_candidates(
            model, tokenizer, prompt, num_candidates,
            temperature=temperature, top_p=top_p, device=device
        )
        
        # Parse and score each candidate
        cands = []
        for cand_text in candidate_texts:
            parsed_json, parse_ok = parse_json_response(cand_text)
            
            if parse_ok:
                parse_success += 1
            total_candidates += 1
            
            # Calculate reward if we have target
            reward = 0.0
            if target and parse_ok:
                reward = reward_calc.calculate_ternary_reward(
                    predicted=parsed_json,
                    gold=target
                )
            
            cands.append({
                'text': cand_text,
                'parsed_json': parsed_json,
                'parse_ok': parse_ok,
                'reward': reward
            })
        
        candidates_data.append({
            'id': example.get('id', ''),
            'prompt': prompt,
            'target_json': target,
            'candidates': cands
        })
        
        # Also save aggregate scores
        scores_data.append({
            'id': example.get('id', ''),
            'parse_rate': sum(c['parse_ok'] for c in cands) / len(cands),
            'mean_reward': sum(c['reward'] for c in cands) / len(cands),
            'max_reward': max(c['reward'] for c in cands),
            'min_reward': min(c['reward'] for c in cands)
        })
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidates_file = output_dir / 'candidates.jsonl'
    scores_file = output_dir / 'scores.jsonl'
    
    save_json_data(candidates_data, candidates_file)
    save_json_data(scores_data, scores_file)
    
    # Calculate statistics
    overall_parse_rate = parse_success / total_candidates if total_candidates > 0 else 0
    mean_reward = sum(s['mean_reward'] for s in scores_data) / len(scores_data) if scores_data else 0
    
    logger.info(f"✅ Generated {len(candidates_data)} examples × {num_candidates} candidates")
    logger.info(f"   Overall parse rate: {overall_parse_rate:.2%}")
    logger.info(f"   Mean reward: {mean_reward:.3f}")
    
    # Save manifest
    manifest = {
        'policy_ckpt': str(policy_ckpt),
        'data_file': str(data_file),
        'num_examples': len(candidates_data),
        'num_candidates': num_candidates,
        'temperature': temperature,
        'top_p': top_p,
        'overall_parse_rate': overall_parse_rate,
        'mean_reward': mean_reward,
        'outputs': {
            'candidates': str(candidates_file),
            'scores': str(scores_file)
        }
    }
    save_manifest(manifest, output_dir / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description="Generate K candidates per prompt for RL training")
    parser.add_argument("--policy_ckpt", type=str, required=True,
                       help="Path to SFT checkpoint (model directory)")
    parser.add_argument("--data_file", type=str, default="datasets/finqa_processed/val.jsonl",
                       help="Input data file (JSONL)")
    parser.add_argument("--split", type=str, default="val", choices=['train', 'val', 'test'],
                       help="Which split to use (will use datasets/finqa_processed/{split}.jsonl)")
    parser.add_argument("--output_dir", type=str, default="outputs/run_001/04_candidates",
                       help="Output directory for candidates")
    parser.add_argument("--num_candidates", type=int, default=4,
                       help="Number of candidates to generate per prompt (K)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling top-p")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit number of examples (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Determine data file from split if not explicitly provided
    if args.data_file == "datasets/finqa_processed/val.jsonl" and args.split != "val":
        args.data_file = f"datasets/finqa_processed/{args.split}.jsonl"
    
    logger.info("="*70)
    logger.info("🎲 Candidate Generation for RL Training")
    logger.info("="*70)
    logger.info(f"Policy: {args.policy_ckpt}")
    logger.info(f"Data: {args.data_file}")
    logger.info(f"K candidates: {args.num_candidates}")
    logger.info(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    logger.info("="*70)
    
    process_dataset(
        data_file=Path(args.data_file),
        policy_ckpt=Path(args.policy_ckpt),
        output_dir=Path(args.output_dir),
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_samples=args.max_samples,
        device=args.device
    )
    
    logger.info("✅ Candidate generation complete!")


if __name__ == "__main__":
    main()
