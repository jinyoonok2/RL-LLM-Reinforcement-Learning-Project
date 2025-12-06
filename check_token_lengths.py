#!/usr/bin/env python3
"""Check actual token lengths in FinQA dataset to optimize max_length"""

import json
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Check both train and val
datasets = {
    'train': 'datasets/finqa_with_rewards/train.jsonl',
    'val': 'datasets/finqa_with_rewards/val.jsonl'
}

all_lengths = []

for split_name, data_file in datasets.items():
    data_path = Path(data_file)
    
    if not data_path.exists():
        print(f"⚠️  {data_file} not found, skipping")
        continue
    
    with open(data_path) as f:
        examples = [json.loads(line) for line in f]
    
    print(f"\n{'='*60}")
    print(f"Analyzing {split_name.upper()} set: {len(examples)} examples")
    print(f"{'='*60}")
    
    lengths = []
    
    for ex in examples:
        question = ex['question']
        candidates = ex.get('candidates', [])[:8]
        
        for cand in candidates:
            answer = cand.get('answer', '')
            program = cand.get('program', '')
            
            # Same formatting as in training code
            if isinstance(program, list) and program:
                program_str = str(program[0]) if program else ''
            else:
                program_str = str(program) if program else ''
            
            # Format exactly as in training
            if program_str:
                text = f"Question: {question}\nProgram: {program_str}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
            
            # Tokenize
            tokens = tokenizer(text, return_tensors='pt')
            length = tokens['input_ids'].shape[1]
            lengths.append(length)
    
    lengths = np.array(lengths)
    all_lengths.extend(lengths)
    
    print(f"\n📊 Token Length Statistics:")
    print(f"   Total sequences: {len(lengths):,}")
    print(f"   Mean: {lengths.mean():.1f} tokens")
    print(f"   Median: {np.median(lengths):.1f} tokens")
    print(f"   Std Dev: {lengths.std():.1f} tokens")
    print(f"   Min: {lengths.min()} tokens")
    print(f"   Max: {lengths.max()} tokens")
    print(f"\n📈 Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(lengths, p)
        print(f"   {p:5.1f}%: {val:6.1f} tokens")
    
    # Check truncation at different max_lengths
    print(f"\n✂️  Truncation Analysis:")
    for max_len in [128, 256, 384, 512, 768, 1024]:
        truncated = (lengths > max_len).sum()
        pct = 100 * truncated / len(lengths)
        if pct > 0:
            print(f"   max_length={max_len:4d}: {truncated:5d} truncated ({pct:5.2f}%)")
        else:
            print(f"   max_length={max_len:4d}: ✅ No truncation")

# Overall summary
print(f"\n{'='*60}")
print(f"OVERALL SUMMARY")
print(f"{'='*60}")

all_lengths = np.array(all_lengths)
print(f"\n📊 Combined Statistics:")
print(f"   Total sequences: {len(all_lengths):,}")
print(f"   Mean: {all_lengths.mean():.1f} tokens")
print(f"   Median: {np.median(all_lengths):.1f} tokens")
print(f"   99th percentile: {np.percentile(all_lengths, 99):.1f} tokens")
print(f"   Max: {all_lengths.max()} tokens")

print(f"\n💡 Recommendations:")
if np.percentile(all_lengths, 99.9) <= 256:
    print(f"   ✅ max_length=256 is EXCELLENT (covers 99.9% of data)")
    print(f"   📉 Could speed up training by 2x vs max_length=512")
elif np.percentile(all_lengths, 99) <= 256:
    print(f"   ✅ max_length=256 is GOOD (covers 99% of data)")
    print(f"   ⚠️  Truncates {(all_lengths > 256).sum()} sequences (1%)")
    print(f"   📉 Could speed up training by 2x vs max_length=512")
elif np.percentile(all_lengths, 95) <= 256:
    print(f"   ⚠️  max_length=256 is RISKY (truncates 5% of data)")
    print(f"   💡 Consider max_length=384 as compromise")
elif np.percentile(all_lengths, 99.9) <= 512:
    print(f"   ✅ max_length=512 is SAFE (covers 99.9% of data)")
    print(f"   ⚠️  Truncates {(all_lengths > 512).sum()} sequences ({100*(all_lengths > 512).sum()/len(all_lengths):.2f}%)")
elif np.percentile(all_lengths, 99) <= 512:
    print(f"   ✅ max_length=512 is GOOD (covers 99% of data)")
    print(f"   ⚠️  Truncates {(all_lengths > 512).sum()} sequences ({100*(all_lengths > 512).sum()/len(all_lengths):.1f}%)")
else:
    print(f"   ⚠️  max_length=512 truncates {100*(all_lengths > 512).sum()/len(all_lengths):.1f}% of data")
    print(f"   💡 Consider max_length={int(np.percentile(all_lengths, 99))} for 99% coverage")

print(f"\n🎯 Current setting: max_length=512")
truncated_512 = (all_lengths > 512).sum()
if truncated_512 == 0:
    print(f"   ✅ Perfect! No sequences truncated")
else:
    print(f"   ⚠️  Truncates {truncated_512:,} sequences ({100*truncated_512/len(all_lengths):.2f}%)")
    lost_tokens = (all_lengths[all_lengths > 512] - 512).sum()
    print(f"   📊 Total tokens lost to truncation: {lost_tokens:,}")

