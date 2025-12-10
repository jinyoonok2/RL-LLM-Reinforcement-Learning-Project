#!/usr/bin/env python3
"""
Evaluate and compare models (SFT vs RL algorithms)
Tests how well models select high-reward candidates from the test set.

Usage:
    # Compare SFT vs GRPO
    python evaluate_models.py --models outputs/run_001/04_sft_llama3b/best_model outputs/run_001/06_grpo/best_model
    
    # Evaluate single model
    python evaluate_models.py --models outputs/run_001/04_sft_llama3b/best_model
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.common import setup_logging

logger = setup_logging()


class CandidateRankingModel(nn.Module):
    """Model that ranks candidates."""
    
    def __init__(self, base_model, num_candidates: int = 8):
        super().__init__()
        self.base_model = base_model
        self.num_candidates = num_candidates
        self.hidden_size = base_model.config.hidden_size
        
        dtype = next(base_model.parameters()).dtype
        
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1, dtype=dtype)
        )
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0] // self.num_candidates
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_indices, sequence_lengths]
        
        if pooled.device != next(self.score_head.parameters()).device:
            pooled = pooled.to(next(self.score_head.parameters()).device)
        
        scores = self.score_head(pooled)
        scores = scores.view(batch_size, self.num_candidates)
        
        return scores


class FinQATestDataset(Dataset):
    """Test dataset for evaluation."""
    
    def __init__(self, data_file: Path, tokenizer, max_length: int = 256, num_candidates: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_candidates = num_candidates
        
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        logger.info(f"Loading test data from {data_file}")
        with open(data_file, 'r') as f:
            self.examples = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(self.examples)} test examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        question = ex['question']
        candidates = ex['candidates'][:self.num_candidates]
        
        while len(candidates) < self.num_candidates:
            candidates.append({
                'answer': '',
                'reward': 0.0,
                'is_gold': False
            })
        
        input_texts = []
        rewards = []
        
        for cand in candidates:
            answer = cand.get('answer', '')
            program = cand.get('program', '')
            
            if isinstance(program, list) and program:
                program_str = str(program[0]) if program else ''
            else:
                program_str = str(program) if program else ''
            
            if program_str:
                text = f"Question: {question}\nProgram: {program_str}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
            
            input_texts.append(text)
            rewards.append(float(cand.get('reward', 0.0)))
        
        encodings = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'rewards': torch.tensor(rewards, dtype=torch.float),
            'example_id': ex.get('id', ''),
            'question': question
        }


def collate_fn(batch):
    """Collate batch."""
    input_ids = torch.cat([x['input_ids'] for x in batch], dim=0)
    attention_mask = torch.cat([x['attention_mask'] for x in batch], dim=0)
    rewards = torch.stack([x['rewards'] for x in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'rewards': rewards,
        'example_id': [x['example_id'] for x in batch],
        'question': [x['question'] for x in batch]
    }


def load_model(checkpoint_path: str, base_model: str = "meta-llama/Llama-3.2-3B", device: str = "cuda"):
    """Load a trained model checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer_path = Path(checkpoint_path)
    if (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    if adapter_config_path.exists():
        logger.info("Loading LoRA checkpoint")
        base = AutoModel.from_pretrained(base_model, torch_dtype=dtype)
        base = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        raise FileNotFoundError(f"No adapter_config.json found at {checkpoint_path}")
    
    model = CandidateRankingModel(base, num_candidates=8)
    
    # Load score_head
    score_head_path = Path(checkpoint_path) / "score_head.pt"
    if score_head_path.exists():
        score_head_state = torch.load(score_head_path, map_location='cpu')
        model.score_head.load_state_dict(score_head_state)
    else:
        logger.warning(f"No score_head.pt found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    return model, tokenizer


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cuda") -> Dict:
    """Evaluate a model on test set."""
    model.eval()
    
    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_examples = 0
    
    total_selected_reward = 0.0
    total_max_reward = 0.0
    total_mean_reward = 0.0
    
    all_predictions = []
    all_rewards = []
    all_selected_rewards = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        rewards = batch['rewards'].to(device)
        
        # Get model predictions
        scores = model(input_ids, attention_mask)
        
        # Get top predictions
        top_indices = scores.topk(5, dim=1).indices
        
        # Best candidate (ground truth)
        best_idx = rewards.argmax(dim=1)
        
        # Top-1 accuracy
        predicted_idx = scores.argmax(dim=1)
        correct = (predicted_idx == best_idx).sum().item()
        total_correct += correct
        
        # Top-3 and Top-5 accuracy
        for i in range(len(best_idx)):
            if best_idx[i] in top_indices[i, :3]:
                total_top3 += 1
            if best_idx[i] in top_indices[i, :5]:
                total_top5 += 1
        
        total_examples += len(predicted_idx)
        
        # Reward metrics
        batch_indices = torch.arange(len(predicted_idx), device=device)
        selected_rewards = rewards[batch_indices, predicted_idx]
        max_rewards = rewards.max(dim=1).values
        mean_rewards = rewards.mean(dim=1)
        
        total_selected_reward += selected_rewards.sum().item()
        total_max_reward += max_rewards.sum().item()
        total_mean_reward += mean_rewards.sum().item()
        
        # Store for analysis
        all_predictions.extend(predicted_idx.cpu().numpy())
        all_rewards.append(rewards.cpu().numpy())
        all_selected_rewards.extend(selected_rewards.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': total_correct / total_examples,
        'top3_accuracy': total_top3 / total_examples,
        'top5_accuracy': total_top5 / total_examples,
        'avg_selected_reward': total_selected_reward / total_examples,
        'avg_max_reward': total_max_reward / total_examples,
        'avg_mean_reward': total_mean_reward / total_examples,
        'reward_ratio': total_selected_reward / total_max_reward if total_max_reward > 0 else 0,
        'total_examples': total_examples,
        'all_selected_rewards': all_selected_rewards,
        'all_rewards': np.concatenate(all_rewards, axis=0)
    }
    
    return metrics


def print_comparison(results: Dict[str, Dict]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Header
    models = list(results.keys())
    print(f"\n{'Metric':<30}", end="")
    for model_name in models:
        print(f"{model_name:<20}", end="")
    print()
    print("-" * (30 + 20 * len(models)))
    
    # Metrics to compare
    metrics_to_show = [
        ('accuracy', 'Top-1 Accuracy', '{:.2%}'),
        ('top3_accuracy', 'Top-3 Accuracy', '{:.2%}'),
        ('top5_accuracy', 'Top-5 Accuracy', '{:.2%}'),
        ('avg_selected_reward', 'Avg Selected Reward', '{:.4f}'),
        ('avg_max_reward', 'Avg Max Reward', '{:.4f}'),
        ('reward_ratio', 'Reward Ratio', '{:.2%}'),
    ]
    
    for metric_key, metric_name, fmt in metrics_to_show:
        print(f"{metric_name:<30}", end="")
        for model_name in models:
            value = results[model_name][metric_key]
            print(fmt.format(value).rjust(20), end="")
        print()
    
    print("-" * (30 + 20 * len(models)))
    print(f"{'Total Examples':<30}", end="")
    for model_name in models:
        print(f"{results[model_name]['total_examples']:<20}", end="")
    print()
    print("="*80)
    
    # Show improvements
    if len(models) > 1:
        baseline = models[0]
        print(f"\nIMPROVEMENTS OVER {baseline.upper()}:")
        print("-" * 80)
        for model_name in models[1:]:
            print(f"\n{model_name}:")
            acc_diff = (results[model_name]['accuracy'] - results[baseline]['accuracy']) * 100
            reward_diff = (results[model_name]['reward_ratio'] - results[baseline]['reward_ratio']) * 100
            print(f"  Accuracy: {acc_diff:+.2f}% points")
            print(f"  Reward Ratio: {reward_diff:+.2f}% points")


def plot_comparison(results: Dict[str, Dict], output_dir: Path):
    """Plot comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(results.keys())
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    top1 = [results[m]['accuracy'] * 100 for m in models]
    top3 = [results[m]['top3_accuracy'] * 100 for m in models]
    top5 = [results[m]['top5_accuracy'] * 100 for m in models]
    
    ax.bar(x - width, top1, width, label='Top-1')
    ax.bar(x, top3, width, label='Top-3')
    ax.bar(x + width, top5, width, label='Top-5')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Reward comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    selected = [results[m]['avg_selected_reward'] for m in models]
    maximum = [results[m]['avg_max_reward'] for m in models]
    
    ax.bar(x - width/2, selected, width, label='Selected Reward')
    ax.bar(x + width/2, maximum, width, label='Max Possible Reward')
    
    ax.set_ylabel('Average Reward')
    ax.set_title('Model Reward Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Reward distribution
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(models):
        selected_rewards = results[model_name]['all_selected_rewards']
        axes[i].hist(selected_rewards, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{model_name}\nSelected Reward Distribution')
        axes[i].set_xlabel('Reward')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(np.mean(selected_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(selected_rewards):.4f}')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="Paths to model checkpoints to evaluate")
    parser.add_argument("--model_names", type=str, nargs='+',
                       help="Names for models (default: use directory names)")
    parser.add_argument("--data_dir", type=str, 
                       default="datasets/finqa_with_rewards",
                       help="Path to data directory")
    parser.add_argument("--base_model", type=str,
                       default="meta-llama/Llama-3.2-3B",
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--test_split", type=str, default="test.jsonl",
                       help="Test split filename")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Model names
    if args.model_names:
        model_names = args.model_names
    else:
        model_names = [Path(p).parent.name for p in args.models]
    
    # Load test data (use first model's tokenizer)
    logger.info("Loading test dataset...")
    _, tokenizer = load_model(args.models[0], args.base_model, device)
    
    test_dataset = FinQATestDataset(
        Path(args.data_dir) / args.test_split,
        tokenizer,
        max_length=256,
        num_candidates=8
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate each model
    results = {}
    for model_path, model_name in zip(args.models, model_names):
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")
        
        model, _ = load_model(model_path, args.base_model, device)
        metrics = evaluate_model(model, test_loader, device)
        results[model_name] = metrics
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Reward Ratio: {metrics['reward_ratio']:.2%}")
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON (without numpy arrays)
    results_json = {
        model_name: {k: v for k, v in metrics.items() 
                    if k not in ['all_selected_rewards', 'all_rewards']}
        for model_name, metrics in results.items()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {output_dir / 'results.json'}")
    
    # Plot comparison
    plot_comparison(results, output_dir)
    
    logger.info("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
