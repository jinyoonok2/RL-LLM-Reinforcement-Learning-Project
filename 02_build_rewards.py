#!/usr/bin/env python3
"""
FinQA Reward Function Builder
Implements configurable, composable reward functions for RL training.

This module provides the reward scoring system that all RL methods (PPO, GRPO, RLOO, DPO)
will use to evaluate model outputs. It integrates with the ternary reward calculator from
01_prepare_dataset.py and adds additional reward components.

Usage:
    python 02_build_rewards.py --schema configs/schema.json --output_dir outputs/finqa_rl/02_rewards
    python 02_build_rewards.py --weights "exact=1.0,program=0.3,format=0.2" --run_tests
"""

import argparse
import json
import logging
import yaml
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ternary reward calculator from preprocessing module
# Note: Python module names can't start with numbers, so we import directly
sys.path.insert(0, str(Path(__file__).parent))

# Try to import from the preprocessing module
TernaryRewardCalculator = None
try:
    # Import the entire module and extract the class
    spec = importlib.util.spec_from_file_location(
        "prepare_dataset",
        Path(__file__).parent / "01_prepare_dataset.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TernaryRewardCalculator = module.TernaryRewardCalculator
        logger.info("✅ Successfully imported TernaryRewardCalculator from 01_prepare_dataset.py")
except Exception as e:
    logger.warning(f"Could not import TernaryRewardCalculator: {e}")
    logger.info("Will use embedded reward calculator")


@dataclass
class RewardComponents:
    """Individual reward components for detailed analysis."""
    exact_match: float = 0.0
    numerical_close: float = 0.0
    program_valid: float = 0.0
    format_valid: float = 0.0
    total: float = 0.0
    
    # Metadata for debugging
    explanation: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class RewardConfig:
    """Configuration for reward function weights and tolerances."""
    # Component weights (should sum to ~1.0 for interpretability)
    weight_exact: float = 1.0
    weight_numerical: float = 0.8
    weight_program: float = 0.3
    weight_format: float = 0.2
    
    # Tolerances for numerical comparison
    numerical_tolerance: float = 0.05  # 5% relative difference
    percentage_tolerance: float = 0.02  # 2% for percentages
    
    # Penalties
    penalty_format_error: float = -0.5
    penalty_empty_prediction: float = -1.0
    
    # Bonuses
    bonus_perfect: float = 0.1  # Extra credit for perfect answers
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def from_weights_string(cls, weights_str: str) -> 'RewardConfig':
        """Parse weights from command-line string like 'exact=1.0,program=0.3'."""
        config = cls()
        for item in weights_str.split(','):
            key, value = item.strip().split('=')
            key = f'weight_{key.strip()}'
            if hasattr(config, key):
                setattr(config, key, float(value))
        return config


class FinQARewardFunction:
    """
    Configurable reward function for FinQA task.
    
    Computes multi-component rewards including:
    - Exact answer matching (with ternary logic)
    - Numerical tolerance checking
    - Program/reasoning validation
    - Format compliance (JSON structure)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Initialize ternary reward calculator if available
        if TernaryRewardCalculator is not None:
            self.ternary_calculator = TernaryRewardCalculator()
        else:
            logger.warning("Using embedded reward calculator (TernaryRewardCalculator not imported)")
            self.ternary_calculator = None
        
        logger.info(f"Initialized FinQARewardFunction with config: {self.config}")
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Remove common formatting
        answer = answer.strip().lower()
        answer = re.sub(r'[,$%\s]', '', answer)
        answer = re.sub(r'\.0+$', '', answer)
        
        return answer
    
    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        if not text:
            return None
            
        cleaned = re.sub(r'[,$%\s]', '', str(text))
        number_match = re.search(r'-?\d+\.?\d*', cleaned)
        
        if number_match:
            try:
                return float(number_match.group())
            except ValueError:
                pass
        return None
    
    def check_exact_match(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Check for exact match using ternary reward logic.
        
        Returns:
            (score, explanation)
        """
        # Use ternary calculator if available
        if self.ternary_calculator:
            # Ternary returns: +1 (exact), 0 (partial), -1 (wrong)
            # We'll convert this to [0, 1] range for composability
            ternary_score, explanation = self.ternary_calculator.calculate_reward(
                prediction, ground_truth, ""  # Question not needed for basic matching
            )
            
            # Convert ternary [-1, 0, 1] to [0, 0.5, 1]
            if ternary_score == 1:
                return (1.0, explanation)
            elif ternary_score == 0:
                return (0.5, explanation)
            else:
                return (0.0, explanation)
        
        # Fallback: simple normalization
        pred_norm = self.normalize_answer(prediction)
        truth_norm = self.normalize_answer(ground_truth)
        
        if pred_norm == truth_norm:
            return (1.0, "exact_match")
        return (0.0, "no_match")
    
    def check_numerical_close(self, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Check if numerical answers are close within tolerance.
        
        Returns:
            (score, explanation) - score in [0, 1]
        """
        pred_num = self.extract_number(prediction)
        truth_num = self.extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            return (0.0, "non_numerical")
        
        if truth_num == 0:
            if abs(pred_num) < 0.001:
                return (1.0, "close_to_zero")
            return (0.0, "not_close_to_zero")
        
        # Calculate relative difference
        relative_diff = abs(pred_num - truth_num) / abs(truth_num)
        
        # Determine tolerance based on percentage presence
        tolerance = (self.config.percentage_tolerance 
                    if '%' in ground_truth 
                    else self.config.numerical_tolerance)
        
        if relative_diff <= tolerance:
            score = 1.0 - (relative_diff / tolerance) * 0.2  # Scale: [1.0, 0.8]
            return (score, f"numerical_close_diff_{relative_diff:.4f}")
        
        # Partial credit for being somewhat close
        if relative_diff <= tolerance * 3:
            score = 0.5 * (1.0 - relative_diff / (tolerance * 3))
            return (score, f"numerical_somewhat_close_diff_{relative_diff:.4f}")
        
        return (0.0, f"numerical_far_diff_{relative_diff:.4f}")
    
    def check_program_valid(self, predicted_program: List[str], 
                           ground_truth_program: List[str]) -> Tuple[float, str]:
        """
        Validate program/reasoning steps.
        
        Returns:
            (score, explanation) - score in [0, 1]
        """
        # Handle missing programs
        if not predicted_program and not ground_truth_program:
            return (1.0, "both_empty_programs")
        
        if not predicted_program:
            return (0.0, "missing_predicted_program")
        
        if not ground_truth_program:
            return (0.5, "no_ground_truth_program")
        
        # Check program structure validity
        try:
            # Basic validation: each step should be parseable
            pred_steps = [step.strip() for step in predicted_program if step.strip()]
            gt_steps = [step.strip() for step in ground_truth_program if step.strip()]
            
            if not pred_steps:
                return (0.0, "empty_predicted_steps")
            
            # Compare step counts (prefer similar length)
            len_diff = abs(len(pred_steps) - len(gt_steps))
            length_score = max(0.0, 1.0 - len_diff * 0.1)
            
            # Check for exact program match
            if pred_steps == gt_steps:
                return (1.0, "exact_program_match")
            
            # Check for partial step matches
            matching_steps = sum(1 for p, g in zip(pred_steps, gt_steps) if p == g)
            match_ratio = matching_steps / max(len(pred_steps), len(gt_steps))
            
            # Combined score
            score = (length_score * 0.3 + match_ratio * 0.7)
            return (score, f"partial_program_match_ratio_{match_ratio:.2f}")
            
        except Exception as e:
            return (0.0, f"program_validation_error_{str(e)[:50]}")
    
    def check_format_valid(self, prediction: str, 
                          expected_format: str = "json") -> Tuple[float, str]:
        """
        Check if prediction follows expected format.
        
        Returns:
            (score, explanation) - score in [0, 1]
        """
        if expected_format == "json":
            try:
                # Try to parse as JSON
                parsed = json.loads(prediction)
                
                # Check for expected fields
                if isinstance(parsed, dict):
                    has_answer = 'answer' in parsed
                    has_program = 'program' in parsed
                    
                    if has_answer and has_program:
                        return (1.0, "valid_json_complete")
                    elif has_answer:
                        return (0.8, "valid_json_missing_program")
                    else:
                        return (0.5, "valid_json_incomplete")
                
                return (0.6, "valid_json_unexpected_structure")
                
            except json.JSONDecodeError as e:
                return (0.0, f"invalid_json_{str(e)[:30]}")
        
        # For non-JSON formats, just check if non-empty
        if prediction and prediction.strip():
            return (1.0, "non_empty_text")
        
        return (0.0, "empty_prediction")
    
    def calculate(self, 
                 prediction: str,
                 ground_truth: str,
                 question: str = "",
                 predicted_program: Optional[List[str]] = None,
                 ground_truth_program: Optional[List[str]] = None,
                 format_check: bool = False) -> RewardComponents:
        """
        Calculate comprehensive reward for a prediction.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Correct answer
            question: Question text (for context-aware scoring)
            predicted_program: Model's reasoning steps
            ground_truth_program: Correct reasoning steps
            format_check: Whether to check JSON format
        
        Returns:
            RewardComponents with detailed breakdown
        """
        components = RewardComponents()
        
        # 1. Check exact match
        exact_score, exact_exp = self.check_exact_match(prediction, ground_truth)
        components.exact_match = exact_score * self.config.weight_exact
        
        # 2. Check numerical closeness (if not exact match)
        numerical_score = 0.0
        numerical_exp = ""
        if exact_score < 1.0:
            numerical_score, numerical_exp = self.check_numerical_close(prediction, ground_truth)
            components.numerical_close = numerical_score * self.config.weight_numerical
        
        # 3. Check program validity
        if predicted_program is not None or ground_truth_program is not None:
            program_score, program_exp = self.check_program_valid(
                predicted_program or [], ground_truth_program or []
            )
            components.program_valid = program_score * self.config.weight_program
        else:
            program_exp = "no_program_provided"
        
        # 4. Check format validity
        if format_check:
            format_score, format_exp = self.check_format_valid(prediction)
            components.format_valid = format_score * self.config.weight_format
        else:
            format_exp = "format_not_checked"
        
        # Calculate total reward
        components.total = (
            components.exact_match +
            components.numerical_close +
            components.program_valid +
            components.format_valid
        )
        
        # Apply bonuses/penalties
        if exact_score == 1.0 and (predicted_program == ground_truth_program):
            components.total += self.config.bonus_perfect
        
        if not prediction or not prediction.strip():
            components.total += self.config.penalty_empty_prediction
        
        # Build explanation
        components.explanation = f"exact:{exact_exp}|num:{numerical_exp}|prog:{program_exp}|fmt:{format_exp}"
        components.details = {
            'exact_score': exact_score,
            'numerical_score': numerical_score,
            'prediction': prediction[:100],
            'ground_truth': ground_truth[:100]
        }
        
        return components
    
    def calculate_batch(self, 
                       predictions: List[str],
                       ground_truths: List[str],
                       questions: Optional[List[str]] = None,
                       predicted_programs: Optional[List[List[str]]] = None,
                       ground_truth_programs: Optional[List[List[str]]] = None) -> List[RewardComponents]:
        """
        Calculate rewards for a batch of predictions.
        
        Returns:
            List of RewardComponents
        """
        if questions is None:
            questions = [""] * len(predictions)
        if predicted_programs is None:
            predicted_programs = [None] * len(predictions)
        if ground_truth_programs is None:
            ground_truth_programs = [None] * len(predictions)
        
        results = []
        for pred, gt, q, pred_prog, gt_prog in zip(
            predictions, ground_truths, questions, predicted_programs, ground_truth_programs
        ):
            reward = self.calculate(pred, gt, q, pred_prog, gt_prog)
            results.append(reward)
        
        return results


def create_unit_tests() -> List[Dict[str, Any]]:
    """
    Create unit test cases for reward function validation.
    
    Returns:
        List of test cases with inputs and expected outcomes
    """
    tests = [
        {
            "name": "exact_match_simple",
            "prediction": "123.45",
            "ground_truth": "123.45",
            "expected_exact": 1.0,
            "description": "Simple exact numerical match"
        },
        {
            "name": "exact_match_with_formatting",
            "prediction": "$123.45",
            "ground_truth": "123.45",
            "expected_exact": 1.0,
            "description": "Match after normalization"
        },
        {
            "name": "numerical_close",
            "prediction": "123.45",
            "ground_truth": "123.50",
            "expected_exact": 0.5,  # Ternary gives partial credit
            "expected_numerical": 0.8,  # 0.8 weight applied
            "description": "Within 5% tolerance (ternary partial + numerical)"
        },
        {
            "name": "percentage_close",
            "prediction": "15.5%",
            "ground_truth": "15.6%",
            "expected_exact": 0.5,  # Ternary partial credit
            "expected_numerical": 0.75,  # ~0.94 * 0.8 weight
            "description": "Percentage within 2% tolerance"
        },
        {
            "name": "empty_prediction",
            "prediction": "",
            "ground_truth": "123",
            "expected_exact": 0.0,
            "expect_penalty": True,
            "description": "Empty prediction should be penalized"
        },
        {
            "name": "empty_ground_truth",
            "prediction": "some answer",
            "ground_truth": "",
            "expected_exact": 0.5,  # Partial credit via ternary
            "description": "Empty ground truth gets partial credit"
        },
        {
            "name": "program_exact_match",
            "prediction": "100",
            "ground_truth": "100",
            "predicted_program": ["add(50, 50)", "result"],
            "ground_truth_program": ["add(50, 50)", "result"],
            "expected_program": 0.3,  # 1.0 * 0.3 weight
            "description": "Exact program match (with 0.3 weight)"
        },
        {
            "name": "program_partial_match",
            "prediction": "100",
            "ground_truth": "100",
            "predicted_program": ["add(50, 50)", "result"],
            "ground_truth_program": ["add(40, 60)", "result"],
            "expected_program": 0.19,  # ~0.64 * 0.3 weight
            "description": "Partial program match (50% steps + length penalty)"
        },
        {
            "name": "format_valid_json",
            "prediction": '{"answer": "123", "program": ["step1"]}',
            "ground_truth": "123",
            "check_format": True,
            "expected_format": 0.2,  # 1.0 * 0.2 weight
            "description": "Valid JSON format (with 0.2 weight)"
        },
        {
            "name": "format_invalid_json",
            "prediction": "{invalid json",
            "ground_truth": "123",
            "check_format": True,
            "expected_format": 0.0,
            "description": "Invalid JSON format"
        }
    ]
    
    return tests


def run_unit_tests(reward_fn: FinQARewardFunction, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run unit tests and report results.
    
    Returns:
        Test results summary
    """
    results = {
        'total': len(tests),
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    print("\n" + "="*70)
    print("🧪 Running Reward Function Unit Tests")
    print("="*70 + "\n")
    
    for test in tests:
        test_name = test['name']
        prediction = test['prediction']
        ground_truth = test['ground_truth']
        
        # Calculate reward
        reward = reward_fn.calculate(
            prediction=prediction,
            ground_truth=ground_truth,
            question=test.get('question', ''),
            predicted_program=test.get('predicted_program'),
            ground_truth_program=test.get('ground_truth_program'),
            format_check=test.get('check_format', False)
        )
        
        # Check expectations
        passed = True
        failures = []
        
        if 'expected_exact' in test:
            if abs(reward.exact_match - test['expected_exact']) > 0.1:
                passed = False
                failures.append(f"exact_match: got {reward.exact_match:.2f}, expected {test['expected_exact']}")
        
        if 'expected_numerical' in test:
            if abs(reward.numerical_close - test['expected_numerical']) > 0.1:
                passed = False
                failures.append(f"numerical_close: got {reward.numerical_close:.2f}, expected {test['expected_numerical']}")
        
        if 'expected_program' in test:
            if abs(reward.program_valid - test['expected_program']) > 0.1:
                passed = False
                failures.append(f"program_valid: got {reward.program_valid:.2f}, expected {test['expected_program']}")
        
        if 'expected_format' in test:
            if abs(reward.format_valid - test['expected_format']) > 0.1:
                passed = False
                failures.append(f"format_valid: got {reward.format_valid:.2f}, expected {test['expected_format']}")
        
        if 'expect_penalty' in test and test['expect_penalty']:
            if reward.total >= 0:
                passed = False
                failures.append(f"total: got {reward.total:.2f}, expected negative")
        
        # Record result
        status = "✅ PASS" if passed else "❌ FAIL"
        results['passed' if passed else 'failed'] += 1
        
        print(f"{status} {test_name}")
        print(f"  Description: {test['description']}")
        print(f"  Prediction: '{prediction}' | Ground Truth: '{ground_truth}'")
        print(f"  Rewards: exact={reward.exact_match:.2f}, num={reward.numerical_close:.2f}, "
              f"prog={reward.program_valid:.2f}, fmt={reward.format_valid:.2f}, total={reward.total:.2f}")
        
        if not passed:
            print(f"  ⚠️  Failures: {', '.join(failures)}")
        
        print()
        
        results['details'].append({
            'test_name': test_name,
            'passed': passed,
            'failures': failures,
            'reward': asdict(reward)
        })
    
    print("="*70)
    print(f"Test Summary: {results['passed']}/{results['total']} passed")
    print("="*70 + "\n")
    
    return results


def save_reward_spec(config: RewardConfig, output_dir: Path):
    """Save reward specification to YAML file."""
    spec = {
        'reward_config': asdict(config),
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'description': 'FinQA reward function specification with ternary logic',
        'components': {
            'exact_match': 'Uses ternary reward logic: +1 (exact), 0 (partial), -1 (wrong)',
            'numerical_close': 'Tolerates 5% relative difference for numbers, 2% for percentages',
            'program_valid': 'Validates reasoning steps against ground truth',
            'format_valid': 'Checks JSON format compliance'
        }
    }
    
    output_path = output_dir / 'reward_spec.yaml'
    with open(output_path, 'w') as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✅ Saved reward specification to {output_path}")


def save_manifest(output_dir: Path, test_results: Optional[Dict] = None):
    """Save module manifest with metadata."""
    manifest = {
        'module': '02_build_rewards',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'test_results': test_results
    }
    
    output_path = output_dir / 'manifest.json'
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Saved manifest to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FinQA reward functions")
    parser.add_argument("--output_dir", type=str, default="outputs/finqa_rl/02_rewards",
                       help="Output directory for reward specs and tests")
    parser.add_argument("--weights", type=str, default="exact=1.0,numerical=0.8,program=0.3,format=0.2",
                       help="Reward component weights (comma-separated)")
    parser.add_argument("--schema", type=str, default="configs/schema.json",
                       help="JSON schema file (optional)")
    parser.add_argument("--run_tests", action="store_true",
                       help="Run unit tests")
    parser.add_argument("--save_tests", action="store_true",
                       help="Save unit test cases to file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Building FinQA Reward Function")
    logger.info(f"Output directory: {output_dir}")
    
    # Create reward configuration
    config = RewardConfig.from_weights_string(args.weights)
    logger.info(f"Reward weights: {args.weights}")
    
    # Initialize reward function
    reward_fn = FinQARewardFunction(config)
    
    # Save reward specification
    save_reward_spec(config, output_dir)
    
    # Create and optionally run unit tests
    tests = create_unit_tests()
    
    if args.save_tests:
        tests_dir = output_dir / 'unit_tests'
        tests_dir.mkdir(exist_ok=True)
        
        for i, test in enumerate(tests):
            test_file = tests_dir / f"test_{i:03d}_{test['name']}.json"
            with open(test_file, 'w') as f:
                json.dump(test, f, indent=2)
        
        logger.info(f"✅ Saved {len(tests)} unit tests to {tests_dir}")
    
    test_results = None
    if args.run_tests:
        test_results = run_unit_tests(reward_fn, tests)
    
    # Save manifest
    save_manifest(output_dir, test_results)
    
    print("\n🎉 Reward function building complete!")
    print(f"📁 Outputs saved to: {output_dir}")
    print(f"📊 Reward spec: {output_dir}/reward_spec.yaml")
    if args.save_tests:
        print(f"🧪 Unit tests: {output_dir}/unit_tests/")
    
    if test_results:
        if test_results['failed'] > 0:
            print(f"\n⚠️  {test_results['failed']} tests failed. Review output above.")
            return 1
        else:
            print(f"\n✅ All {test_results['passed']} tests passed!")
    
    print("\n💡 Next steps:")
    print(f"  1. Review reward spec: cat {output_dir}/reward_spec.yaml")
    print(f"  2. Test on real data: python 03_sft_train.py --reward_spec {output_dir}/reward_spec.yaml")
    
    return 0


if __name__ == "__main__":
    exit(main())
