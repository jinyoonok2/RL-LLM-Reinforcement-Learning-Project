#!/usr/bin/env python3
"""
FinQA Dataset Validation - Quick validation of dataset files.

Usage:
    python 00_check_data.py --data_root datasets/finqa
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

from utils.common import setup_logging, print_section

logger = setup_logging()


def check_finqa_dataset(data_root: str) -> bool:
    """
    Validate FinQA dataset files.
    
    Args:
        data_root: Path to FinQA dataset directory
        
    Returns:
        True if validation passes, False otherwise
    """
    data_root = Path(data_root)
    required_files = ['train.json', 'val.json', 'test.json']
    required_fields = {'id', 'pre_text', 'post_text', 'table', 'qa'}
    
    print_section("FinQA Dataset Validation")
    
    # 1. Check files exist
    print("📁 Checking files...")
    missing = []
    for filename in required_files:
        filepath = data_root / filename
        if not filepath.exists():
            missing.append(filename)
            print(f"  ❌ Missing: {filename}")
        else:
            print(f"  ✅ Found: {filename}")
    
    if missing:
        print(f"\n❌ Missing required files: {missing}")
        return False
    
    # 2. Load and validate JSON
    print("\n📖 Loading and validating...")
    data = {}
    stats = {}
    
    for filename in required_files:
        filepath = data_root / filename
        split_name = filename.replace('.json', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            # Basic validation
            if not isinstance(examples, list):
                print(f"  ❌ {filename}: Expected list, got {type(examples)}")
                return False
            
            # Check schema
            malformed = 0
            empty_answers = 0
            answer_types = Counter()
            
            for i, ex in enumerate(examples[:100]):  # Sample first 100
                if not isinstance(ex, dict):
                    malformed += 1
                    continue
                
                missing_fields = required_fields - set(ex.keys())
                if missing_fields:
                    malformed += 1
                    if i < 5:  # Only log first few
                        print(f"  ⚠️  Example {i}: Missing {missing_fields}")
                
                # Check answer
                if 'qa' in ex and isinstance(ex['qa'], dict):
                    answer = ex['qa'].get('answer', '')
                    if not answer or answer == '':
                        empty_answers += 1
                    elif isinstance(answer, (int, float)):
                        answer_types['numeric'] += 1
                    elif '%' in str(answer):
                        answer_types['percentage'] += 1
                    else:
                        answer_types['text'] += 1
            
            data[split_name] = examples
            stats[split_name] = {
                'total': len(examples),
                'malformed': malformed,
                'empty_answers': empty_answers,
                'answer_types': dict(answer_types)
            }
            
            print(f"  ✅ {filename}: {len(examples):,} examples")
            if malformed > 0:
                print(f"     ⚠️  {malformed} malformed examples")
            if empty_answers > 0:
                print(f"     ⚠️  {empty_answers} empty answers")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ {filename}: Invalid JSON - {e}")
            return False
        except Exception as e:
            print(f"  ❌ {filename}: Error - {e}")
            return False
    
    # 3. Print statistics
    print("\n📊 Dataset Statistics:")
    total = sum(s['total'] for s in stats.values())
    print(f"  Total examples: {total:,}")
    
    for split, s in stats.items():
        print(f"\n  {split.upper()}:")
        print(f"    Examples: {s['total']:,} ({s['total']/total*100:.1f}%)")
        if s['answer_types']:
            for atype, count in s['answer_types'].items():
                print(f"    {atype}: {count} ({count/s['total']*100:.1f}%)")
    
    # 4. Show samples
    print("\n🔍 Sample Examples:")
    for split, examples in list(data.items())[:1]:  # Just show train
        for i, ex in enumerate(examples[:2]):
            qa = ex.get('qa', {})
            print(f"\n  Example {i+1}:")
            print(f"    Q: {qa.get('question', 'N/A')[:80]}...")
            print(f"    A: {qa.get('answer', 'N/A')}")
            print(f"    Program steps: {len(qa.get('program', []))}")
    
    print_section("✅ Validation Complete", char='-')
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate FinQA dataset")
    parser.add_argument("--data_root", type=str, default="datasets/finqa",
                       help="Path to FinQA dataset directory")
    args = parser.parse_args()
    
    try:
        success = check_finqa_dataset(args.data_root)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
