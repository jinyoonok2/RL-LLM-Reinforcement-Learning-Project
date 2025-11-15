#!/usr/bin/env python3
"""
Debug script to test SFT model generation and reward calculation.
"""

import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.rewards import FinQARewardCalculator

def main():
    print("=" * 80)
    print("SFT Model Debug Tool")
    print("=" * 80)
    
    # Load model
    model_path = "outputs/run_001/03_sft/ckpt_sft/best"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run training first or specify different checkpoint")
        return
    
    print(f"\n1. Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print("   ✅ Model loaded")
    
    # Load validation data
    val_file = "datasets/finqa_processed/val.jsonl"
    print(f"\n2. Loading validation data from: {val_file}")
    examples = []
    with open(val_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Just load 3 examples
                break
            examples.append(json.loads(line))
    print(f"   ✅ Loaded {len(examples)} examples")
    
    # Initialize reward calculator
    print("\n3. Initializing reward calculator")
    reward_calc = FinQARewardCalculator()
    print("   ✅ Reward calculator ready")
    
    # Test each example
    print("\n4. Testing model generation:")
    print("=" * 80)
    
    for i, example in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        
        # Show input
        input_text = example['input_text']
        question = example.get('question', '')
        print(f"\n📝 INPUT TEXT (first 300 chars):")
        print(input_text[:300] + "...")
        print(f"\n❓ QUESTION: {question}")
        
        # Create prompt (same as training)
        prompt = f"{input_text}\n\nQuestion: {question}\n\nProvide your answer in JSON format with 'answer' and 'program' fields:\n"
        
        print(f"\n🎯 FULL PROMPT (last 200 chars):")
        print("..." + prompt[-200:])
        
        # Generate
        print(f"\n🤖 GENERATING...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n📤 GENERATED OUTPUT:")
        print(generated_text)
        
        # Parse JSON
        print(f"\n🔍 PARSING JSON:")
        try:
            parsed = json.loads(generated_text)
            print(f"   ✅ Valid JSON!")
            print(f"   Keys: {list(parsed.keys())}")
            print(f"   Answer: {parsed.get('answer', 'MISSING')}")
            print(f"   Program: {parsed.get('program', 'MISSING')}")
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON Parse Error: {e}")
            parsed = None
        
        # Calculate reward
        print(f"\n💰 REWARD CALCULATION:")
        target_answer = example.get('target_answer', '')
        target_program = example.get('target_program', '')
        print(f"   Ground truth answer: {target_answer}")
        print(f"   Ground truth program: {target_program}")
        
        if parsed:
            reward_result = reward_calc.calculate_component_rewards(
                prediction=str(parsed.get('answer', '')),
                ground_truth=str(target_answer),
                predicted_program=parsed.get('program', ''),
                ground_truth_program=target_program
            )
            reward = reward_result.total
            print(f"   Exact match: {reward_result.exact_match:.4f}")
            print(f"   Numerical close: {reward_result.numerical_close:.4f}")
            print(f"   Program valid: {reward_result.program_valid:.4f}")
            print(f"   Format valid: {reward_result.format_valid:.4f}")
        else:
            reward = -0.2  # Format penalty
        
        print(f"   TOTAL REWARD: {reward:.4f}")
        
        if reward == 0:
            print(f"\n   ⚠️  ZERO REWARD - Debugging:")
            if not parsed:
                print(f"      - JSON parsing failed")
            else:
                print(f"      - Predicted answer: '{parsed.get('answer', '')}'")
                print(f"      - Ground truth: '{target_answer}'")
                print(f"      - Match: {parsed.get('answer', '') == target_answer}")
    
    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
