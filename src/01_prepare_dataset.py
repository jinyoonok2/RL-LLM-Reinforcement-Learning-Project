#!/usr/bin/env python3
"""
FinQA Dataset Preprocessing with Ternary Reward Method
Preprocesses FinQA dataset for RL training with sophisticated reward assignment.

The Ternary Reward Method addresses the challenge of empty/missing answers:
- +1: Exact match with ground truth
- 0: Reasonable attempt (empty answers, near-misses, partial correctness) 
- -1: Completely incorrect or nonsensical

Usage:
    python src/01_prepare_dataset.py --input_dir datasets/finqa --output_dir datasets/finqa_processed
    python src/01_prepare_dataset.py --input_dir datasets/finqa --output_dir datasets/finqa_processed --model_name microsoft/DialoGPT-medium
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    from transformers import AutoTokenizer, AutoConfig
    import torch
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install transformers torch")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedExample:
    """Structured representation of a processed FinQA example."""
    id: str
    question: str
    context: str  # Combined pre_text + table + post_text
    ground_truth_answer: str
    ground_truth_program: List[str]
    input_text: str  # Formatted for model input
    target_text: str  # Expected model output
    reward_type: str  # 'exact', 'partial', 'impossible'
    metadata: Dict[str, Any]


class TernaryRewardCalculator:
    """Calculates ternary rewards for FinQA answers."""
    
    def __init__(self):
        self.numerical_tolerance = 0.05  # 5% tolerance for numerical answers
        self.percentage_tolerance = 0.02  # 2% tolerance for percentages
        
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Remove common formatting
        answer = answer.strip().lower()
        answer = re.sub(r'[,$%\s]', '', answer)  # Remove commas, $, %, spaces
        answer = re.sub(r'\.0+$', '', answer)  # Remove trailing zeros
        
        return answer
    
    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        if not text:
            return None
            
        # Remove common currency/percentage symbols
        cleaned = re.sub(r'[,$%\s]', '', str(text))
        
        # Try to extract number
        number_match = re.search(r'-?\d+\.?\d*', cleaned)
        if number_match:
            try:
                return float(number_match.group())
            except ValueError:
                pass
        return None
    
    def is_numerical_close(self, pred: str, target: str) -> bool:
        """Check if two numerical answers are close within tolerance."""
        pred_num = self.extract_number(pred)
        target_num = self.extract_number(target)
        
        if pred_num is None or target_num is None:
            return False
        
        if target_num == 0:
            return abs(pred_num) < 0.001  # Very small threshold for zero
        
        # Check relative difference
        relative_diff = abs(pred_num - target_num) / abs(target_num)
        tolerance = self.percentage_tolerance if '%' in target else self.numerical_tolerance
        
        return relative_diff <= tolerance
    
    def classify_question_type(self, question: str) -> str:
        """Classify question type to determine appropriate reward strategy."""
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ['did', 'does', 'is', 'are', 'was', 'were']):
            return 'yes_no'
        elif any(phrase in question_lower for phrase in ['why', 'how', 'explain', 'describe']):
            return 'qualitative'
        elif any(phrase in question_lower for phrase in ['what', 'which', 'how much', 'how many']):
            return 'quantitative'
        else:
            return 'other'
    
    def calculate_reward(self, predicted: str, ground_truth: str, question: str) -> Tuple[int, str]:
        """
        Calculate ternary reward for prediction.
        
        Returns:
            Tuple of (reward_value, explanation)
            reward_value: +1 (correct), 0 (partial), -1 (wrong)
        """
        pred_norm = self.normalize_answer(predicted)
        truth_norm = self.normalize_answer(ground_truth)
        question_type = self.classify_question_type(question)
        
        # Handle empty ground truth (the problematic cases we identified)
        if not ground_truth or ground_truth.strip() == '':
            if question_type in ['yes_no', 'qualitative']:
                # For yes/no and qualitative questions, empty GT means partial credit
                return (0, "empty_ground_truth_qualitative")
            else:
                # For quantitative questions, empty GT suggests unanswerable
                return (0, "empty_ground_truth_unanswerable")
        
        # Handle empty prediction
        if not predicted or predicted.strip() == '':
            return (-1, "empty_prediction")
        
        # Exact match (highest reward)
        if pred_norm == truth_norm:
            return (1, "exact_match")
        
        # Numerical near-miss (partial reward)
        if self.is_numerical_close(predicted, ground_truth):
            return (0, "numerical_close")
        
        # For qualitative questions, give partial credit for reasonable attempts
        if question_type == 'qualitative' and len(predicted.strip()) > 10:
            return (0, "qualitative_attempt")
        
        # For yes/no questions, check for reasonable binary answers
        if question_type == 'yes_no':
            pred_lower = predicted.lower().strip()
            if pred_lower in ['yes', 'no', 'true', 'false', '1', '0']:
                return (0, "binary_attempt")
        
        # Everything else is wrong
        return (-1, "incorrect")


class FinQAPreprocessor:
    """Main preprocessor for FinQA dataset with ternary reward support."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.reward_calculator = TernaryRewardCalculator()
        self.stats = defaultdict(int)
        
        # Load tokenizer
        try:
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = {
                'pad_token': '<pad>',
                'sep_token': '<sep>',
                'additional_special_tokens': ['<question>', '<context>', '<answer>', '<program>']
            }
            
            self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"✅ Tokenizer loaded with vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def format_table(self, table_data: List[Dict[str, Any]]) -> str:
        """Convert table data to readable text format."""
        if not table_data or not isinstance(table_data, list):
            return ""
        
        # Extract headers (keys from first row)
        if table_data and isinstance(table_data[0], dict):
            headers = list(table_data[0].keys())
            
            # Create formatted table
            table_lines = [" | ".join(headers)]
            table_lines.append("-" * len(table_lines[0]))
            
            for row in table_data:
                if isinstance(row, dict):
                    values = [str(row.get(header, "")) for header in headers]
                    table_lines.append(" | ".join(values))
            
            return "\n".join(table_lines)
        
        return str(table_data)
    
    def create_context(self, example: Dict[str, Any]) -> str:
        """Create comprehensive context from pre_text, table, and post_text."""
        context_parts = []
        
        # Add pre-text
        pre_text = example.get('pre_text', '')
        if pre_text:
            if isinstance(pre_text, list):
                context_parts.append(" ".join(pre_text))
            else:
                context_parts.append(str(pre_text))
        
        # Add table
        table = example.get('table', [])
        if table:
            table_text = self.format_table(table)
            if table_text:
                context_parts.append(f"Table:\n{table_text}")
        
        # Add post-text
        post_text = example.get('post_text', '')
        if post_text:
            if isinstance(post_text, list):
                context_parts.append(" ".join(post_text))
            else:
                context_parts.append(str(post_text))
        
        return "\n\n".join(context_parts)
    
    def create_input_text(self, question: str, context: str) -> str:
        """Format input for the model."""
        # Create structured input format
        input_parts = [
            "<question>",
            question.strip(),
            "<context>", 
            context.strip()
        ]
        
        return " ".join(input_parts)
    
    def create_target_text(self, answer: str, program: List[str]) -> str:
        """Format target output for the model."""
        target_parts = ["<answer>", str(answer).strip()]
        
        if program and isinstance(program, list):
            target_parts.extend(["<program>"] + [str(step) for step in program])
        
        return " ".join(target_parts)
    
    def process_example(self, example: Dict[str, Any]) -> Optional[ProcessedExample]:
        """Process a single FinQA example."""
        try:
            # Extract basic info
            example_id = example.get('id', 'unknown')
            qa = example.get('qa', {})
            
            question = qa.get('question', '').strip()
            if not question:
                self.stats['skipped_no_question'] += 1
                return None
            
            # Get answer and program
            answer = qa.get('answer', '')
            program = qa.get('program', [])
            
            # Create context
            context = self.create_context(example)
            
            # Create input and target texts
            input_text = self.create_input_text(question, context)
            target_text = self.create_target_text(answer, program)
            
            # Calculate reward type (for future reward assignment)
            if answer and answer.strip():
                reward_type = 'exact'  # Has ground truth for exact matching
            else:
                reward_type = 'partial'  # Empty answer, use partial reward strategy
            
            # Create metadata
            metadata = {
                'question_type': self.reward_calculator.classify_question_type(question),
                'has_answer': bool(answer and answer.strip()),
                'has_program': bool(program),
                'context_length': len(context),
                'input_tokens': len(self.tokenizer.encode(input_text)) if self.tokenizer else 0,
                'target_tokens': len(self.tokenizer.encode(target_text)) if self.tokenizer else 0,
                'original_table_rows': len(example.get('table', [])),
                'pre_text_length': len(str(example.get('pre_text', ''))),
                'post_text_length': len(str(example.get('post_text', '')))
            }
            
            # Update statistics
            self.stats['processed'] += 1
            self.stats[f'question_type_{metadata["question_type"]}'] += 1
            self.stats[f'reward_type_{reward_type}'] += 1
            
            if metadata['has_answer']:
                self.stats['has_answer'] += 1
            else:
                self.stats['empty_answer'] += 1
            
            return ProcessedExample(
                id=example_id,
                question=question,
                context=context,
                ground_truth_answer=answer,
                ground_truth_program=program,
                input_text=input_text,
                target_text=target_text,
                reward_type=reward_type,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
            self.stats['error'] += 1
            return None
    
    def process_split(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Process a single data split."""
        logger.info(f"Processing {input_file.name}...")
        
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        processed_examples = []
        
        for i, example in enumerate(examples):
            if i % 1000 == 0:
                logger.info(f"  Processing example {i}/{len(examples)}")
            
            processed = self.process_example(example)
            if processed:
                processed_examples.append(processed)
        
        # Convert to serializable format
        output_data = {
            'examples': [
                {
                    'id': ex.id,
                    'question': ex.question,
                    'context': ex.context,
                    'ground_truth_answer': ex.ground_truth_answer,
                    'ground_truth_program': ex.ground_truth_program,
                    'input_text': ex.input_text,
                    'target_text': ex.target_text,
                    'reward_type': ex.reward_type,
                    'metadata': ex.metadata
                }
                for ex in processed_examples
            ],
            'split_stats': {
                'total_input': len(examples),
                'total_processed': len(processed_examples),
                'empty_answers': sum(1 for ex in processed_examples if not ex.ground_truth_answer.strip()),
                'has_answers': sum(1 for ex in processed_examples if ex.ground_truth_answer.strip()),
                'avg_input_tokens': sum(ex.metadata['input_tokens'] for ex in processed_examples) / len(processed_examples) if processed_examples else 0,
                'avg_target_tokens': sum(ex.metadata['target_tokens'] for ex in processed_examples) / len(processed_examples) if processed_examples else 0,
            }
        }
        
        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✅ Saved {len(processed_examples)} examples to {output_file}")
        return output_data['split_stats']
    
    def create_reward_examples(self, processed_file: Path, num_examples: int = 100) -> Path:
        """Create examples showing how ternary rewards would be calculated."""
        
        with open(processed_file, 'r') as f:
            data = json.load(f)
        
        examples = data['examples'][:num_examples]
        reward_examples = []
        
        for ex in examples:
            # Simulate some predictions to show reward calculation
            predictions = [
                ex['ground_truth_answer'],  # Perfect match
                ex['ground_truth_answer'].replace('0', '1') if ex['ground_truth_answer'] else 'unknown',  # Slight variation
                '',  # Empty prediction
                'completely wrong answer'  # Wrong answer
            ]
            
            for i, pred in enumerate(predictions):
                reward, explanation = self.reward_calculator.calculate_reward(
                    pred, ex['ground_truth_answer'], ex['question']
                )
                
                reward_examples.append({
                    'id': ex['id'],
                    'question': ex['question'][:100] + '...' if len(ex['question']) > 100 else ex['question'],
                    'ground_truth': ex['ground_truth_answer'],
                    'prediction': pred,
                    'reward': reward,
                    'explanation': explanation,
                    'question_type': ex['metadata']['question_type']
                })
        
        # Save reward examples
        output_file = processed_file.parent / 'reward_examples.json'
        with open(output_file, 'w') as f:
            json.dump(reward_examples, f, indent=2)
        
        logger.info(f"✅ Created {len(reward_examples)} reward examples in {output_file}")
        return output_file
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n📊 Preprocessing Statistics:")
        print("=" * 50)
        print(f"Total processed: {self.stats['processed']:,}")
        print(f"Has answer: {self.stats['has_answer']:,}")
        print(f"Empty answer: {self.stats['empty_answer']:,}")
        print(f"Errors: {self.stats['error']:,}")
        print(f"Skipped (no question): {self.stats['skipped_no_question']:,}")
        
        print(f"\nQuestion Types:")
        for key, value in self.stats.items():
            if key.startswith('question_type_'):
                qtype = key.replace('question_type_', '')
                print(f"  {qtype}: {value:,}")
        
        print(f"\nReward Types:")
        for key, value in self.stats.items():
            if key.startswith('reward_type_'):
                rtype = key.replace('reward_type_', '')
                print(f"  {rtype}: {value:,}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FinQA dataset with ternary reward method")
    parser.add_argument("--input_dir", type=str, default="datasets/finqa", 
                       help="Input directory containing FinQA JSON files")
    parser.add_argument("--output_dir", type=str, default="datasets/finqa_processed",
                       help="Output directory for processed files")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="Model name for tokenizer")
    parser.add_argument("--create_reward_examples", action="store_true",
                       help="Create examples showing reward calculation")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Initialize preprocessor
    try:
        preprocessor = FinQAPreprocessor(model_name=args.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        return 1
    
    logger.info("🚀 Starting FinQA preprocessing with ternary reward method...")
    
    # Process each split
    splits_to_process = {
        'train.json': 'train_processed.json',
        'val.json': 'val_processed.json', 
        'test.json': 'test_processed.json'
    }
    
    # Optional: include private test if it exists
    private_test_path = input_dir / 'private_test.json'
    if private_test_path.exists():
        splits_to_process['private_test.json'] = 'private_test_processed.json'
    
    all_stats = {}
    
    for input_name, output_name in splits_to_process.items():
        input_file = input_dir / input_name
        output_file = output_dir / output_name
        
        if input_file.exists():
            split_stats = preprocessor.process_split(input_file, output_file)
            all_stats[input_name] = split_stats
            
            # Create reward examples for train split
            if args.create_reward_examples and input_name == 'train.json':
                preprocessor.create_reward_examples(output_file)
        else:
            logger.warning(f"Skipping missing file: {input_file}")
    
    # Save overall statistics
    stats_file = output_dir / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'model_name': args.model_name,
            'split_stats': all_stats,
            'global_stats': dict(preprocessor.stats),
            'ternary_reward_info': {
                'description': 'Ternary reward method: +1 (exact), 0 (partial), -1 (wrong)',
                'empty_answer_strategy': 'Assign partial reward (0) for empty ground truth answers',
                'numerical_tolerance': preprocessor.reward_calculator.numerical_tolerance,
                'percentage_tolerance': preprocessor.reward_calculator.percentage_tolerance
            }
        }, f, indent=2)
    
    # Print summary
    preprocessor.print_statistics()
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"📁 Processed files saved to: {output_dir}")
    print(f"📊 Statistics saved to: {stats_file}")
    
    if args.create_reward_examples:
        print(f"🎯 Reward examples saved to: {output_dir}/reward_examples.json")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Review reward examples: cat {output_dir}/reward_examples.json")
    print(f"  2. Start training: python src/03_sft_train.py --data_dir {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())