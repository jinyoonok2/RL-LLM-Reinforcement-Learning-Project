#!/usr/bin/env python3
"""
Generate Multiple Candidates per Question for Classification RL Training
Creates a pool of candidate answers by perturbing gold answers programmatically.

Usage:
    python 02_generate_candidates.py --input_dir datasets/finqa_processed --output_dir datasets/finqa_candidates
    python 02_generate_candidates.py --num_candidates 8 --corruption_rate 0.5
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

from utils.common import setup_logging, save_manifest, save_json_data, load_json_data

logger = setup_logging()


class CandidateGenerator:
    """Generate diverse candidate answers through programmatic perturbations."""
    
    def __init__(self, num_candidates: int = 8, corruption_rate: float = 0.5, seed: int = 42):
        """
        Args:
            num_candidates: Total candidates to generate per question (including gold)
            corruption_rate: Fraction of candidates that should be corrupted (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.num_candidates = num_candidates
        self.corruption_rate = corruption_rate
        random.seed(seed)
        
        # Arithmetic operations for program perturbation
        self.operations = ['add', 'subtract', 'multiply', 'divide']
        self.op_symbols = {
            'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'
        }
    
    def generate_candidates(self, example: Dict) -> List[Dict]:
        """
        Generate multiple candidate answers for a single question.
        
        Returns list of candidates, each with:
            - answer: answer string
            - program: program steps (if applicable)
            - is_gold: whether this is the gold answer
            - corruption_type: type of perturbation applied (if any)
        """
        candidates = []
        target_answer = example['target_answer']
        target_program = example.get('target_program', [])
        
        # Candidate 1: Gold answer (always first)
        candidates.append({
            'answer': target_answer,
            'program': target_program,
            'is_gold': True,
            'corruption_type': None
        })
        
        # Determine how many corrupted vs similar candidates
        num_corrupted = max(1, int((self.num_candidates - 1) * self.corruption_rate))
        num_similar = (self.num_candidates - 1) - num_corrupted
        
        # Generate similar (plausible but possibly wrong) candidates
        for _ in range(num_similar):
            candidate = self._generate_similar_candidate(target_answer, target_program)
            candidates.append(candidate)
        
        # Generate corrupted (obviously wrong) candidates
        for _ in range(num_corrupted):
            candidate = self._generate_corrupted_candidate(target_answer, target_program)
            candidates.append(candidate)
        
        # Shuffle candidates (except gold stays at index 0)
        gold = candidates[0]
        others = candidates[1:]
        random.shuffle(others)
        
        return [gold] + others
    
    def _generate_similar_candidate(self, answer: str, program: List) -> Dict:
        """Generate plausible but potentially incorrect candidate."""
        strategies = [
            self._perturb_numerical_answer,
            self._perturb_program_order,
            self._perturb_program_constants,
            self._add_formatting_variation
        ]
        
        # Try a random strategy
        strategy = random.choice(strategies)
        try:
            perturbed_answer, perturbed_program, corruption_type = strategy(answer, program)
            return {
                'answer': perturbed_answer,
                'program': perturbed_program,
                'is_gold': False,
                'corruption_type': corruption_type
            }
        except:
            # Fallback: just return slightly modified answer
            return {
                'answer': self._add_noise_to_number(answer),
                'program': program,
                'is_gold': False,
                'corruption_type': 'numerical_noise'
            }
    
    def _generate_corrupted_candidate(self, answer: str, program: List) -> Dict:
        """Generate obviously wrong candidate."""
        strategies = [
            self._wrong_operation,
            self._missing_steps,
            self._random_answer,
            self._format_error
        ]
        
        strategy = random.choice(strategies)
        try:
            corrupted_answer, corrupted_program, corruption_type = strategy(answer, program)
            return {
                'answer': corrupted_answer,
                'program': corrupted_program,
                'is_gold': False,
                'corruption_type': corruption_type
            }
        except:
            # Fallback: random number
            return {
                'answer': str(random.randint(1, 10000)),
                'program': [],
                'is_gold': False,
                'corruption_type': 'random'
            }
    
    # === Perturbation Strategies ===
    
    def _perturb_numerical_answer(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Add small numerical noise to answer."""
        new_answer = self._add_noise_to_number(answer)
        return new_answer, program, 'numerical_perturbation'
    
    def _add_noise_to_number(self, answer: str) -> str:
        """Add ±5-15% noise to a numerical answer."""
        # Extract number from answer
        match = re.search(r'-?\d+\.?\d*', answer)
        if match:
            num = float(match.group())
            # Add random noise
            noise = random.uniform(0.05, 0.15) * num
            if random.random() < 0.5:
                noise = -noise
            new_num = num + noise
            
            # Format appropriately
            if '.' in match.group() or abs(new_num) < 1:
                new_str = f"{new_num:.2f}"
            else:
                new_str = str(int(new_num))
            
            # Replace in original answer
            return answer.replace(match.group(), new_str)
        return answer
    
    def _perturb_program_order(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Shuffle program steps (may produce different answer)."""
        if len(program) <= 1:
            return answer, program, 'no_perturbation'
        
        new_program = program.copy()
        random.shuffle(new_program)
        
        # Try to recalculate answer from shuffled program
        new_answer = self._execute_program(new_program)
        if new_answer is None:
            new_answer = answer  # Keep original if execution fails
        
        return new_answer, new_program, 'program_order'
    
    def _perturb_program_constants(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Modify constants in program steps."""
        if not program:
            return answer, program, 'no_perturbation'
        
        new_program = deepcopy(program)
        
        # Randomly modify one constant in the program
        for step in new_program:
            if isinstance(step, str):
                # Look for numbers in the step
                numbers = re.findall(r'\d+\.?\d*', step)
                if numbers:
                    old_num = random.choice(numbers)
                    new_num = float(old_num) * random.uniform(0.8, 1.2)
                    new_num_str = f"{new_num:.2f}" if '.' in old_num else str(int(new_num))
                    step = step.replace(old_num, new_num_str, 1)
        
        new_answer = self._execute_program(new_program)
        if new_answer is None:
            new_answer = self._add_noise_to_number(answer)
        
        return new_answer, new_program, 'program_constants'
    
    def _add_formatting_variation(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Add/remove formatting (%, $, commas, etc.)."""
        variations = []
        
        # Extract number
        match = re.search(r'-?\d+\.?\d*', answer)
        if match:
            num = match.group()
            
            # Different format variations
            if '%' in answer:
                variations.append(num)  # Remove %
            else:
                variations.append(f"{num}%")  # Add %
            
            if '$' in answer:
                variations.append(num)  # Remove $
            else:
                variations.append(f"${num}")  # Add $
            
            # Add/remove commas
            if ',' in num:
                variations.append(num.replace(',', ''))
            elif len(num) > 3:
                # Add comma formatting
                try:
                    formatted = f"{int(float(num)):,}"
                    variations.append(formatted)
                except:
                    pass
        
        if variations:
            return random.choice(variations), program, 'formatting'
        return answer, program, 'no_perturbation'
    
    def _wrong_operation(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Replace operation with wrong one."""
        if not program:
            return str(random.randint(1, 1000)), [], 'wrong_operation'
        
        new_program = deepcopy(program)
        
        # Find and replace an operation
        for i, step in enumerate(new_program):
            if isinstance(step, str):
                for op in self.operations:
                    if op in step.lower():
                        # Replace with different operation
                        wrong_ops = [o for o in self.operations if o != op]
                        new_op = random.choice(wrong_ops)
                        new_program[i] = step.replace(op, new_op)
                        break
        
        new_answer = self._execute_program(new_program)
        if new_answer is None:
            new_answer = str(random.randint(1, 10000))
        
        return new_answer, new_program, 'wrong_operation'
    
    def _missing_steps(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Remove steps from program."""
        if len(program) <= 1:
            return "", [], 'missing_steps'
        
        # Remove random steps
        num_to_remove = random.randint(1, max(1, len(program) // 2))
        new_program = random.sample(program, len(program) - num_to_remove)
        
        new_answer = self._execute_program(new_program)
        if new_answer is None:
            new_answer = "incomplete"
        
        return new_answer, new_program, 'missing_steps'
    
    def _random_answer(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Generate completely random answer."""
        random_num = random.randint(1, 100000)
        return str(random_num), [], 'random'
    
    def _format_error(self, answer: str, program: List) -> Tuple[str, List, str]:
        """Introduce formatting errors."""
        corruptions = [
            "N/A",
            "Error",
            "",
            "invalid",
            "###",
            answer + " (maybe)",
            f"approximately {answer}"
        ]
        return random.choice(corruptions), program, 'format_error'
    
    def _execute_program(self, program: List) -> str:
        """
        Attempt to execute a program and return answer.
        Simplified execution - returns None if it fails.
        """
        try:
            # This is a placeholder - actual execution would need proper parsing
            # For now, just return None to indicate we can't execute
            return None
        except:
            return None


def generate_candidates_for_dataset(input_dir: str, output_dir: str, num_candidates: int = 8, 
                                     corruption_rate: float = 0.5, seed: int = 42):
    """
    Generate candidate pools for entire dataset.
    
    Args:
        input_dir: Input directory with prepared dataset (from 01_prepare_dataset.py)
        output_dir: Output directory for candidate datasets
        num_candidates: Number of candidates per question
        corruption_rate: Fraction of corrupted candidates
        seed: Random seed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = CandidateGenerator(num_candidates, corruption_rate, seed)
    
    splits = ['train', 'val', 'test']
    stats = {
        'total_questions': 0,
        'total_candidates': 0,
        'by_split': {},
        'corruption_types': {}
    }
    
    logger.info(f"Generating {num_candidates} candidates per question ({corruption_rate:.0%} corrupted)")
    
    for split in splits:
        input_file = input_dir / f"{split}.jsonl"
        if not input_file.exists():
            logger.warning(f"Skipping {split}: file not found")
            continue
        
        logger.info(f"Processing {split} split...")
        
        # Load prepared dataset
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Generate candidates for each example
        output_examples = []
        corruption_counts = {}
        
        for ex in examples:
            candidates = generator.generate_candidates(ex)
            
            # Count corruption types
            for cand in candidates:
                ctype = cand['corruption_type'] or 'gold'
                corruption_counts[ctype] = corruption_counts.get(ctype, 0) + 1
            
            output_examples.append({
                'id': ex['id'],
                'input_text': ex['input_text'],
                'question': ex['question'],
                'candidates': candidates,
                'metadata': ex.get('metadata', {})
            })
        
        # Save output
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in output_examples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(output_examples)} questions with {len(output_examples) * num_candidates} candidates")
        
        stats['by_split'][split] = {
            'questions': len(output_examples),
            'candidates': len(output_examples) * num_candidates,
            'corruption_types': corruption_counts
        }
        stats['total_questions'] += len(output_examples)
        stats['total_candidates'] += len(output_examples) * num_candidates
        
        # Merge corruption type counts
        for ctype, count in corruption_counts.items():
            stats['corruption_types'][ctype] = stats['corruption_types'].get(ctype, 0) + count
    
    # Save statistics
    stats_file = output_dir / 'candidate_stats.json'
    save_json_data(stats, str(stats_file))
    
    save_manifest(
        output_dir,
        '02_generate_candidates',
        {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'num_candidates': num_candidates,
            'corruption_rate': corruption_rate,
            'seed': seed
        },
        stats
    )
    
    logger.info(f"✅ Candidate generation complete")
    logger.info(f"   Questions: {stats['total_questions']}")
    logger.info(f"   Total candidates: {stats['total_candidates']}")
    logger.info(f"   Corruption types: {stats['corruption_types']}")


def main():
    parser = argparse.ArgumentParser(description="Generate candidate answer pools for classification RL")
    parser.add_argument("--input_dir", type=str, default="datasets/finqa_processed",
                       help="Input directory with prepared dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/finqa_candidates",
                       help="Output directory for candidate datasets")
    parser.add_argument("--num_candidates", type=int, default=8,
                       help="Number of candidates to generate per question")
    parser.add_argument("--corruption_rate", type=float, default=0.5,
                       help="Fraction of candidates that should be corrupted (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    generate_candidates_for_dataset(
        args.input_dir,
        args.output_dir,
        args.num_candidates,
        args.corruption_rate,
        args.seed
    )


if __name__ == "__main__":
    main()
