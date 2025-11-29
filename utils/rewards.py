"""
Unified Reward Calculation for FinQA
Provides reward functions for RL training and evaluation.
"""

import re
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Individual reward components for detailed analysis."""
    exact_match: float = 0.0
    numerical_close: float = 0.0
    program_valid: float = 0.0
    format_valid: float = 0.0
    total: float = 0.0
    explanation: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class FinQARewardCalculator:
    """Unified reward calculator for FinQA - handles both ternary and component-based rewards."""
    
    def __init__(self, numerical_tolerance: float = 0.10, percentage_tolerance: float = 0.05):
        self.numerical_tolerance = numerical_tolerance  # Increased from 0.05 to 0.10
        self.percentage_tolerance = percentage_tolerance  # Increased from 0.02 to 0.05
        
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not isinstance(answer, str):
            answer = str(answer)
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
    
    def is_numerical_close(self, pred: str, target: str) -> bool:
        """Check if two numerical answers are close within tolerance."""
        pred_num = self.extract_number(pred)
        target_num = self.extract_number(target)
        
        if pred_num is None or target_num is None:
            return False
        
        if target_num == 0:
            return abs(pred_num) < 0.001
        
        relative_diff = abs(pred_num - target_num) / abs(target_num)
        tolerance = self.percentage_tolerance if '%' in target else self.numerical_tolerance
        return relative_diff <= tolerance
    
    def calculate_ternary_reward(self, predicted: str, ground_truth: str, question: str = "") -> Tuple[int, str]:
        """
        Calculate ternary reward: +1 (correct), 0 (partial), -1 (wrong).
        Used for dataset preparation.
        """
        pred_norm = self.normalize_answer(predicted)
        truth_norm = self.normalize_answer(ground_truth)
        
        # Empty ground truth
        if not ground_truth or ground_truth.strip() == '':
            return (0, "empty_ground_truth")
        
        # Empty prediction
        if not predicted or predicted.strip() == '':
            return (-1, "empty_prediction")
        
        # Exact match
        if pred_norm == truth_norm:
            return (1, "exact_match")
        
        # Numerical near-miss (more generous scoring)
        if self.is_numerical_close(predicted, ground_truth):
            return (0.5, "numerical_close")  # Better than 0 for close answers
        
        # Partial numerical credit (within 20%)
        pred_num = self.extract_number(predicted)
        truth_num = self.extract_number(ground_truth)
        if pred_num is not None and truth_num is not None:
            if truth_num != 0:
                relative_diff = abs(pred_num - truth_num) / abs(truth_num)
                if relative_diff <= 0.20:  # Within 20%
                    return (0.3, f"partial_numerical_{relative_diff:.2f}")
        
        # Qualitative attempt (if long enough)
        if question and any(w in question.lower() for w in ['why', 'how', 'explain']) and len(predicted.strip()) > 10:
            return (0.2, "qualitative_attempt")
        
        # At least attempted an answer
        if len(predicted.strip()) > 5:
            return (-0.5, "incorrect_attempt")  # Less penalty for trying
        
        return (-1, "incorrect")
    
    def calculate_component_rewards(
        self,
        prediction: str,
        ground_truth: str,
        predicted_program: List[str] = None,
        ground_truth_program: List[str] = None,
        weights: Dict[str, float] = None
    ) -> RewardComponents:
        """
        Calculate detailed component-based rewards for RL training.
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Correct answer
            predicted_program: Optional program steps
            ground_truth_program: Optional ground truth program
            weights: Optional component weights
            
        Returns:
            RewardComponents with detailed scores
        """
        if weights is None:
            weights = {'exact': 1.0, 'numerical': 0.9, 'program': 0.5, 'format': 0.3}  # Higher rewards for partial credit
        
        components = RewardComponents()
        
        # 1. Exact match component
        pred_norm = self.normalize_answer(prediction)
        truth_norm = self.normalize_answer(ground_truth)
        if pred_norm == truth_norm:
            components.exact_match = weights['exact']
            components.explanation = "exact_match"
        
        # 2. Numerical closeness component
        if components.exact_match == 0.0:
            pred_num = self.extract_number(prediction)
            truth_num = self.extract_number(ground_truth)
            if pred_num is not None and truth_num is not None:
                if truth_num == 0:
                    if abs(pred_num) < 0.001:
                        components.numerical_close = weights['numerical']
                else:
                    relative_diff = abs(pred_num - truth_num) / abs(truth_num)
                    tolerance = self.percentage_tolerance if '%' in ground_truth else self.numerical_tolerance
                    if relative_diff <= tolerance:
                        components.numerical_close = weights['numerical'] * (1.0 - relative_diff / tolerance)
                        components.explanation = f"numerical_close_{relative_diff:.4f}"
        
        # 3. Program validity component
        if predicted_program and ground_truth_program:
            try:
                pred_steps = [s.strip() for s in predicted_program if s.strip()]
                truth_steps = [s.strip() for s in ground_truth_program if s.strip()]
                if pred_steps and truth_steps:
                    # Simple overlap scoring
                    matching_steps = sum(1 for s in pred_steps if s in truth_steps)
                    components.program_valid = weights['program'] * (matching_steps / len(truth_steps))
            except:
                pass
        
        # 4. Format validity component (JSON parsing)
        try:
            import json
            json.loads(prediction) if prediction.startswith('{') else None
            components.format_valid = weights['format']
        except:
            pass
        
        # Total reward
        components.total = (
            components.exact_match +
            components.numerical_close +
            components.program_valid +
            components.format_valid
        )
        
        components.details = {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'normalized_pred': pred_norm,
            'normalized_truth': truth_norm
        }
        
        return components
    
    def calculate(self, prediction: str, ground_truth: str, question: str = "", **kwargs) -> RewardComponents:
        """
        Convenience method for compatibility with evaluation code.
        Delegates to calculate_component_rewards().
        """
        return self.calculate_component_rewards(
            prediction=prediction,
            ground_truth=ground_truth,
            predicted_program=kwargs.get('predicted_program'),
            ground_truth_program=kwargs.get('ground_truth_program'),
            weights=kwargs.get('weights')
        )

