#!/usr/bin/env python3
"""
FinQA Reward Function Builder - Configure and test reward functions for RL.

Usage:
    python 02_build_rewards.py --output_dir outputs/run_001/02_rewards
    python 02_build_rewards.py --weights "exact=1.0,numerical=0.8,program=0.3,format=0.2" --run_tests
"""

import argparse
import yaml
from pathlib import Path

from utils.common import setup_logging, save_manifest, print_section
from utils.rewards import FinQARewardCalculator, RewardComponents

logger = setup_logging()


def run_reward_tests(reward_calc: FinQARewardCalculator, weights: dict) -> bool:
    """Run unit tests on reward function."""
    print_section("Running Reward Function Tests")
    
    test_cases = [
        # (prediction, ground_truth, expected_reward_type, description)
        ("1000", "1000", "exact", "Exact match"),
        ("1000", "1001", "numerical_close", "Numerical close"),
        ("50%", "50.5%", "numerical_close", "Percentage close"),
        ("", "1000", "wrong", "Empty prediction"),
        ("1000", "", "partial", "Empty ground truth"),
        ("wrong answer", "1000", "wrong", "Completely wrong"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, truth, expected, desc in test_cases:
        ternary_reward, explanation = reward_calc.calculate_ternary_reward(pred, truth)
        
        # Map ternary to category
        if ternary_reward == 1:
            result = "exact"
        elif ternary_reward == 0:
            result = "partial" if "close" in explanation else "partial"
        else:
            result = "wrong"
        
        if "close" in explanation:
            result = "numerical_close"
        
        status = "✅" if result == expected else "❌"
        print(f"  {status} {desc}")
        print(f"      Pred: '{pred}' | Truth: '{truth}'")
        print(f"      Result: {result} (expected: {expected}) | {explanation}")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Tests: {passed} passed, {failed} failed")
    return failed == 0


def build_reward_spec(output_dir: str, weights: dict, run_tests: bool = False):
    """
    Build and save reward specification.
    
    Args:
        output_dir: Output directory for reward spec
        weights: Component weights dictionary
        run_tests: Whether to run unit tests
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reward_calc = FinQARewardCalculator()
    
    # Run tests if requested
    if run_tests:
        test_passed = run_reward_tests(reward_calc, weights)
        if not test_passed:
            logger.warning("Some tests failed, but continuing...")
    
    # Build reward specification
    reward_spec = {
        'weights': weights,
        'tolerances': {
            'numerical': reward_calc.numerical_tolerance,
            'percentage': reward_calc.percentage_tolerance
        },
        'components': {
            'exact_match': 'Direct string match after normalization',
            'numerical_close': 'Numerical values within tolerance',
            'program_valid': 'Program steps match ground truth',
            'format_valid': 'Valid JSON format'
        },
        'ternary_mapping': {
            '+1': 'exact_match',
            '0': 'partial_credit (numerical_close, qualitative_attempt, etc.)',
            '-1': 'incorrect'
        }
    }
    
    # Save reward spec
    spec_file = output_dir / 'reward_spec.yaml'
    with open(spec_file, 'w') as f:
        yaml.dump(reward_spec, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✅ Saved reward specification to {spec_file}")
    
    # Save manifest
    save_manifest(
        output_dir,
        '02_build_rewards',
        {'weights': weights, 'run_tests': run_tests},
        {'test_passed': test_passed if run_tests else None}
    )
    
    print_section("✅ Reward Function Ready", char='-')


def parse_weights(weights_str: str) -> dict:
    """Parse weights string like 'exact=1.0,numerical=0.8'."""
    weights = {}
    for item in weights_str.split(','):
        key, value = item.strip().split('=')
        weights[key.strip()] = float(value.strip())
    return weights


def main():
    parser = argparse.ArgumentParser(description="Build reward function for RL training")
    parser.add_argument("--output_dir", type=str, default="outputs/run_001/02_rewards",
                       help="Output directory for reward specification")
    parser.add_argument("--weights", type=str, default="exact=1.0,numerical=0.8,program=0.3,format=0.2",
                       help="Component weights (comma-separated key=value pairs)")
    parser.add_argument("--run_tests", action="store_true",
                       help="Run unit tests on reward function")
    args = parser.parse_args()
    
    weights = parse_weights(args.weights)
    build_reward_spec(args.output_dir, weights, args.run_tests)


if __name__ == "__main__":
    main()
