#!/usr/bin/env python3
"""
FinQA Dataset Validation Script
Validates the FinQA dataset format, quality, and provides comprehensive analysis.

Usage:
    python src/00_check_data.py --data_root datasets/finqa
    python src/00_check_data.py --data_root datasets/finqa --detailed
    python src/00_check_data.py --data_root datasets/finqa --sample 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinQADataValidator:
    """Comprehensive FinQA dataset validation and analysis."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.required_files = {
            'train.json': 'training data',
            'val.json': 'validation data', 
            'test.json': 'test data'
        }
        self.optional_files = {
            'private_test.json': 'private test data'
        }
        self.data = {}
        self.stats = {}
        
    def check_file_existence(self) -> bool:
        """Check if all required dataset files exist."""
        logger.info("🔍 Checking file existence...")
        
        missing_files = []
        found_files = []
        
        # Check required files
        for filename, description in self.required_files.items():
            filepath = self.data_root / filename
            if filepath.exists():
                found_files.append(f"✅ {filename} ({description})")
                logger.info(f"Found {filename}: {filepath}")
            else:
                missing_files.append(f"❌ {filename} ({description})")
                logger.error(f"Missing required file: {filepath}")
        
        # Check optional files
        for filename, description in self.optional_files.items():
            filepath = self.data_root / filename
            if filepath.exists():
                found_files.append(f"✅ {filename} ({description}) [optional]")
                logger.info(f"Found optional file {filename}: {filepath}")
        
        # Print summary
        print("\n📁 File Existence Check:")
        for file_info in found_files:
            print(f"  {file_info}")
        
        if missing_files:
            print("\n❌ Missing Files:")
            for file_info in missing_files:
                print(f"  {file_info}")
            return False
        
        print("✅ All required files found!")
        return True
    
    def load_data(self) -> bool:
        """Load and validate JSON files."""
        logger.info("📖 Loading dataset files...")
        
        all_files = {**self.required_files, **self.optional_files}
        
        for filename in all_files.keys():
            filepath = self.data_root / filename
            if not filepath.exists():
                if filename in self.required_files:
                    logger.error(f"Required file missing: {filename}")
                    return False
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data[filename.replace('.json', '')] = data
                    logger.info(f"✅ Loaded {filename}: {len(data)} examples")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in {filename}: {e}")
                return False
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                return False
        
        return True
    
    def validate_schema(self) -> bool:
        """Validate the schema of each dataset split."""
        logger.info("🔬 Validating data schema...")
        
        required_fields = {'id', 'pre_text', 'post_text', 'table', 'qa'}
        qa_required_fields = {'question', 'program', 'answer'}
        
        schema_valid = True
        
        for split_name, examples in self.data.items():
            logger.info(f"Validating {split_name} schema...")
            
            # Private test sets typically don't have answers/programs
            is_private_test = 'private' in split_name.lower()
            
            if not isinstance(examples, list):
                logger.error(f"❌ {split_name}: Data should be a list, got {type(examples)}")
                schema_valid = False
                continue
            
            # Check a sample of examples
            sample_size = min(100, len(examples))
            for i, example in enumerate(examples[:sample_size]):
                if not isinstance(example, dict):
                    logger.error(f"❌ {split_name}[{i}]: Example should be dict, got {type(example)}")
                    schema_valid = False
                    continue
                
                # Check required fields
                missing_fields = required_fields - set(example.keys())
                if missing_fields:
                    logger.error(f"❌ {split_name}[{i}]: Missing fields: {missing_fields}")
                    schema_valid = False
                
                # Check QA structure
                if 'qa' in example and isinstance(example['qa'], dict):
                    if is_private_test:
                        # For private test, only require question
                        if 'question' not in example['qa']:
                            logger.error(f"❌ {split_name}[{i}]: Missing question in QA")
                            schema_valid = False
                    else:
                        # For training/val/test, require all QA fields
                        qa_missing = qa_required_fields - set(example['qa'].keys())
                        if qa_missing:
                            logger.error(f"❌ {split_name}[{i}]: Missing QA fields: {qa_missing}")
                            schema_valid = False
                elif 'qa' in example:
                    logger.error(f"❌ {split_name}[{i}]: 'qa' should be dict, got {type(example['qa'])}")
                    schema_valid = False
        
        if schema_valid:
            print("✅ Schema validation passed!")
        else:
            print("❌ Schema validation failed!")
        
        return schema_valid
    
    def analyze_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        logger.info("📊 Computing dataset statistics...")
        
        stats = {}
        
        for split_name, examples in self.data.items():
            split_stats = {
                'total_examples': len(examples),
                'question_lengths': [],
                'answer_types': Counter(),
                'program_lengths': [],
                'table_sizes': [],
                'text_lengths': {'pre': [], 'post': []},
                'unique_ids': set(),
                'question_types': Counter(),
                'mathematical_operations': Counter()
            }
            
            for example in examples:
                # Basic info
                if 'id' in example:
                    split_stats['unique_ids'].add(example['id'])
                
                # Question analysis
                if 'qa' in example and 'question' in example['qa']:
                    question = example['qa']['question']
                    if isinstance(question, str):
                        split_stats['question_lengths'].append(len(question.split()))
                    
                    # Classify question types
                    question_lower = question.lower()
                    if any(word in question_lower for word in ['what', 'which']):
                        split_stats['question_types']['what/which'] += 1
                    elif any(word in question_lower for word in ['how much', 'how many']):
                        split_stats['question_types']['how much/many'] += 1
                    elif 'percentage' in question_lower or '%' in question:
                        split_stats['question_types']['percentage'] += 1
                    else:
                        split_stats['question_types']['other'] += 1
                
                # Answer analysis
                if 'qa' in example and 'answer' in example['qa']:
                    answer = example['qa']['answer']
                    if isinstance(answer, (int, float)):
                        split_stats['answer_types']['numeric'] += 1
                    elif isinstance(answer, str):
                        if answer.replace('.', '').replace('-', '').isdigit():
                            split_stats['answer_types']['numeric_string'] += 1
                        elif '%' in answer:
                            split_stats['answer_types']['percentage'] += 1
                        else:
                            split_stats['answer_types']['text'] += 1
                
                # Program analysis
                if 'qa' in example and 'program' in example['qa']:
                    program = example['qa']['program']
                    if isinstance(program, list):
                        split_stats['program_lengths'].append(len(program))
                        
                        # Analyze mathematical operations
                        for step in program:
                            if isinstance(step, str):
                                if 'add(' in step:
                                    split_stats['mathematical_operations']['addition'] += 1
                                elif 'subtract(' in step:
                                    split_stats['mathematical_operations']['subtraction'] += 1
                                elif 'multiply(' in step:
                                    split_stats['mathematical_operations']['multiplication'] += 1
                                elif 'divide(' in step:
                                    split_stats['mathematical_operations']['division'] += 1
                
                # Table analysis
                if 'table' in example and isinstance(example['table'], list):
                    split_stats['table_sizes'].append(len(example['table']))
                
                # Text analysis
                if 'pre_text' in example and isinstance(example['pre_text'], str):
                    split_stats['text_lengths']['pre'].append(len(example['pre_text'].split()))
                if 'post_text' in example and isinstance(example['post_text'], str):
                    split_stats['text_lengths']['post'].append(len(example['post_text'].split()))
            
            # Compute summary statistics
            split_stats['unique_id_count'] = len(split_stats['unique_ids'])
            split_stats['avg_question_length'] = (
                sum(split_stats['question_lengths']) / len(split_stats['question_lengths'])
                if split_stats['question_lengths'] else 0
            )
            split_stats['avg_program_length'] = (
                sum(split_stats['program_lengths']) / len(split_stats['program_lengths'])
                if split_stats['program_lengths'] else 0
            )
            split_stats['avg_table_size'] = (
                sum(split_stats['table_sizes']) / len(split_stats['table_sizes'])
                if split_stats['table_sizes'] else 0
            )
            
            stats[split_name] = split_stats
        
        self.stats = stats
        return stats
    
    def print_statistics(self, detailed: bool = False):
        """Print comprehensive statistics."""
        print("\n📊 Dataset Statistics:")
        print("=" * 60)
        
        # Overall summary
        total_examples = sum(stats['total_examples'] for stats in self.stats.values())
        print(f"Total Examples: {total_examples:,}")
        
        # Per-split statistics
        for split_name, stats in self.stats.items():
            print(f"\n📋 {split_name.upper()} Split:")
            print(f"  Examples: {stats['total_examples']:,}")
            print(f"  Unique IDs: {stats['unique_id_count']:,}")
            
            if stats['question_lengths']:
                print(f"  Avg Question Length: {stats['avg_question_length']:.1f} words")
            if stats['program_lengths']:
                print(f"  Avg Program Length: {stats['avg_program_length']:.1f} steps")
            if stats['table_sizes']:
                print(f"  Avg Table Size: {stats['avg_table_size']:.1f} rows")
            
            # Question types
            if stats['question_types']:
                print(f"  Question Types:")
                for qtype, count in stats['question_types'].most_common():
                    percentage = (count / stats['total_examples']) * 100
                    print(f"    - {qtype}: {count} ({percentage:.1f}%)")
            
            # Answer types
            if stats['answer_types']:
                print(f"  Answer Types:")
                for atype, count in stats['answer_types'].most_common():
                    percentage = (count / stats['total_examples']) * 100
                    print(f"    - {atype}: {count} ({percentage:.1f}%)")
            
            if detailed and stats['mathematical_operations']:
                print(f"  Mathematical Operations:")
                for op, count in stats['mathematical_operations'].most_common():
                    print(f"    - {op}: {count}")
    
    def sample_examples(self, n: int = 3):
        """Display sample examples from each split."""
        print(f"\n🔍 Sample Examples (showing {n} per split):")
        print("=" * 60)
        
        for split_name, examples in self.data.items():
            if not examples:
                continue
                
            print(f"\n📝 {split_name.upper()} Samples:")
            sample_size = min(n, len(examples))
            
            for i in range(sample_size):
                example = examples[i]
                print(f"\n  Example {i+1}:")
                print(f"    ID: {example.get('id', 'N/A')}")
                
                if 'qa' in example:
                    qa = example['qa']
                    question = qa.get('question', 'N/A')
                    print(f"    Question: {question[:100]}{'...' if len(question) > 100 else ''}")
                    print(f"    Answer: {qa.get('answer', 'N/A')}")
                    
                    if 'program' in qa and isinstance(qa['program'], list):
                        print(f"    Program Steps: {len(qa['program'])}")
                        if qa['program']:
                            print(f"    First Step: {qa['program'][0]}")
                
                if 'table' in example and isinstance(example['table'], list):
                    print(f"    Table Rows: {len(example['table'])}")
                
                pre_text = example.get('pre_text', '')
                if pre_text:
                    print(f"    Pre-text: {pre_text[:80]}{'...' if len(pre_text) > 80 else ''}")
    
    def check_data_quality(self) -> bool:
        """Perform data quality checks."""
        logger.info("🔍 Performing data quality checks...")
        
        quality_issues = []
        
        for split_name, examples in self.data.items():
            for i, example in enumerate(examples):
                # Check for empty required fields
                if not example.get('id'):
                    quality_issues.append(f"{split_name}[{i}]: Missing or empty ID")
                
                if 'qa' in example:
                    qa = example['qa']
                    if not qa.get('question', '').strip():
                        quality_issues.append(f"{split_name}[{i}]: Empty question")
                    if 'answer' not in qa or qa['answer'] in [None, '', []]:
                        quality_issues.append(f"{split_name}[{i}]: Missing or empty answer")
                    if not qa.get('program'):
                        quality_issues.append(f"{split_name}[{i}]: Missing or empty program")
                
                # Check for duplicate IDs within split
                ids_in_split = [ex.get('id') for ex in examples if ex.get('id')]
                duplicate_ids = [id for id, count in Counter(ids_in_split).items() if count > 1]
                if duplicate_ids:
                    quality_issues.append(f"{split_name}: Duplicate IDs found: {duplicate_ids}")
        
        if quality_issues:
            print(f"\n⚠️  Data Quality Issues Found ({len(quality_issues)}):")
            for issue in quality_issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(quality_issues) > 10:
                print(f"  ... and {len(quality_issues) - 10} more issues")
            return False
        else:
            print("✅ Data quality checks passed!")
            return True
    
    def run_full_validation(self, detailed: bool = False, sample: int = 3) -> bool:
        """Run complete validation pipeline."""
        print("🚀 Starting FinQA Dataset Validation")
        print("=" * 60)
        
        # Step 1: Check file existence
        if not self.check_file_existence():
            return False
        
        # Step 2: Load data
        if not self.load_data():
            return False
        
        # Step 3: Validate schema
        if not self.validate_schema():
            return False
        
        # Step 4: Analyze statistics
        self.analyze_statistics()
        self.print_statistics(detailed=detailed)
        
        # Step 5: Show samples
        if sample > 0:
            self.sample_examples(sample)
        
        # Step 6: Quality checks
        quality_ok = self.check_data_quality()
        
        # Final summary
        print(f"\n🎉 Validation Summary:")
        print("=" * 30)
        print(f"✅ Files: All required files found")
        print(f"✅ Schema: Valid FinQA format")
        print(f"✅ Loading: All files loaded successfully")
        print(f"{'✅' if quality_ok else '⚠️ '} Quality: {'No issues found' if quality_ok else 'Some issues detected'}")
        
        total_examples = sum(len(examples) for examples in self.data.values())
        print(f"\n📊 Ready for training with {total_examples:,} total examples!")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate FinQA dataset")
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="datasets/finqa",
        help="Path to FinQA dataset directory"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Show detailed statistics including mathematical operations"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=3,
        help="Number of sample examples to show per split (0 to disable)"
    )
    
    args = parser.parse_args()
    
    # Create validator and run validation
    validator = FinQADataValidator(args.data_root)
    
    try:
        success = validator.run_full_validation(
            detailed=args.detailed, 
            sample=args.sample
        )
        
        if success:
            logger.info("✅ Dataset validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Dataset validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()