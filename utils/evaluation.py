#!/usr/bin/env python3
"""
Shared Model Evaluation Utilities
Provides reusable evaluation logic for both training validation and final evaluation.

This module is used by:
- Module 03 (SFT training validation)
- Module 10 (final comprehensive evaluation)
- Future RL modules (PPO, GRPO, RLOO evaluation)
"""

import json
import logging
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    avg_reward: float
    parse_rate: float
    num_samples: int
    samples: List[Dict[str, Any]]
    detailed_metrics: Optional[Dict[str, float]] = None


class ModelEvaluator:
    """
    Shared evaluation logic for model validation and testing.
    
    Features:
    - Generate predictions with robust error handling
    - Calculate rewards using reward function
    - Track parse success rate
    - Support both quick validation and comprehensive evaluation
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_fn=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize evaluator.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            reward_fn: Reward function for scoring (optional)
            device: Device to run evaluation on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def generate(self, input_text: str, max_length: int = 512) -> str:
        """
        Generate prediction for input text with robust error handling.
        
        Args:
            input_text: Input prompt
            max_length: Maximum input length
            
        Returns:
            Generated text (empty string on error)
        """
        try:
            # Tokenize and move to GPU
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Use greedy decoding (most stable)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=min(self.max_new_tokens, 64),  # Limit length
                    do_sample=False,  # Greedy (no sampling)
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Decode only the generated part
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return prediction.strip()
        
        except Exception as e:
            logger.warning(f"Generation failed: {str(e)[:100]}, returning empty")
            return ""
    
    def evaluate(
        self,
        dataset,
        num_samples: Optional[int] = None,
        detailed: bool = False,
        description: str = "Evaluation"
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Dataset to evaluate (must have 'examples' attribute)
            num_samples: Number of samples to evaluate (None = all)
            detailed: If True, compute detailed per-component metrics
            description: Description for progress bar
            
        Returns:
            EvaluationResult with metrics and samples
        """
        logger.info(f"📊 Running {description}...")
        self.model.eval()
        
        total_reward = 0
        samples = []
        parse_success = 0
        
        # Determine sample indices
        if num_samples is None:
            indices = list(range(len(dataset)))
        else:
            indices = np.random.choice(
                len(dataset),
                min(num_samples, len(dataset)),
                replace=False
            )
        
        # Component-wise metrics (if detailed)
        if detailed and self.reward_fn:
            component_totals = {
                'exact_match': 0,
                'numerical': 0,
                'program': 0,
                'format': 0
            }
        
        with torch.no_grad():
            for idx in tqdm(indices, desc=description):
                example = dataset[idx]
                
                # Get input text and ground truth
                if hasattr(dataset, 'examples'):
                    input_text = dataset.examples[idx]['input_text']
                    question = dataset.examples[idx]['question']
                else:
                    # Fallback for different dataset structures
                    input_text = example.get('input_text', '')
                    question = example.get('question', '')
                
                ground_truth = example['ground_truth']
                
                # Create prompt with JSON instruction (same as training)
                prompt = f"{input_text}\n\nQuestion: {question}\n\nProvide your answer in JSON format with 'answer' and 'program' fields:\n"
                
                # Generate prediction
                prediction = self.generate(prompt)
                
                # Calculate reward if reward function available
                reward_total = 0
                if self.reward_fn:
                    try:
                        reward = self.reward_fn.calculate(
                            prediction=prediction,
                            ground_truth=ground_truth,
                            question=question
                        )
                        reward_total = reward.total
                        total_reward += reward_total
                        
                        # Track component scores if detailed
                        if detailed:
                            component_totals['exact_match'] += reward.exact_match
                            component_totals['numerical'] += reward.numerical_score
                            component_totals['program'] += reward.program_valid
                            component_totals['format'] += reward.format_score
                    except Exception as e:
                        logger.warning(f"Reward calculation failed for example {idx}: {e}")
                        reward_total = 0
                
                # Check if valid JSON
                parse_ok = False
                try:
                    json.loads(prediction)
                    parse_success += 1
                    parse_ok = True
                except:
                    pass
                
                # Store sample
                sample_data = {
                    'example_id': example.get('example_id', f'sample_{idx}'),
                    'question': question[:100] if detailed else question[:50],
                    'prediction': prediction[:200] if not detailed else prediction,
                    'ground_truth': ground_truth,
                    'reward': reward_total,
                    'parse_ok': parse_ok
                }
                samples.append(sample_data)
        
        # Calculate aggregate metrics
        num_evaluated = len(indices)
        avg_reward = total_reward / num_evaluated if num_evaluated > 0 else 0
        parse_rate = parse_success / num_evaluated if num_evaluated > 0 else 0
        
        # Detailed component metrics
        detailed_metrics = None
        if detailed and self.reward_fn:
            detailed_metrics = {
                'exact_match_rate': component_totals['exact_match'] / num_evaluated,
                'numerical_rate': component_totals['numerical'] / num_evaluated,
                'program_valid_rate': component_totals['program'] / num_evaluated,
                'format_valid_rate': component_totals['format'] / num_evaluated
            }
        
        logger.info(f"{description} Results:")
        logger.info(f"  Samples: {num_evaluated}")
        logger.info(f"  Avg Reward: {avg_reward:.4f}")
        logger.info(f"  Parse Rate: {parse_rate:.2%}")
        
        if detailed_metrics:
            logger.info(f"  Exact Match: {detailed_metrics['exact_match_rate']:.2%}")
            logger.info(f"  Numerical: {detailed_metrics['numerical_rate']:.2%}")
            logger.info(f"  Program Valid: {detailed_metrics['program_valid_rate']:.2%}")
            logger.info(f"  Format Valid: {detailed_metrics['format_valid_rate']:.2%}")
        
        return EvaluationResult(
            avg_reward=avg_reward,
            parse_rate=parse_rate,
            num_samples=num_evaluated,
            samples=samples,
            detailed_metrics=detailed_metrics
        )
