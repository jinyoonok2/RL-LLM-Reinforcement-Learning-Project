#!/usr/bin/env python3
"""
Comprehensive evaluation script that compares all experiment versions
and saves results in multiple formats (CSV, JSON, LaTeX, Markdown).

Usage:
    python evaluate_all_experiments.py
    
This will evaluate all models found in outputs/ and generate comparison reports.
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_models import load_model, FinQATestDataset, collate_fn, evaluate_model
from utils.common import setup_logging
from torch.utils.data import DataLoader

logger = setup_logging()


def find_all_models(base_dir: str = "outputs") -> List[Dict[str, str]]:
    """Find all trained model checkpoints."""
    base_path = Path(base_dir)
    models = []
    
    # Pattern: outputs/run_*/*/best_model
    for checkpoint_dir in base_path.glob("*/*/best_model"):
        # Check if it has adapter files
        if (checkpoint_dir / "adapter_config.json").exists():
            run_name = checkpoint_dir.parent.parent.name  # e.g., run_001
            model_type = checkpoint_dir.parent.name  # e.g., 04_sft_llama3b
            
            # Parse model info
            if "sft" in model_type.lower():
                algo = "SFT"
                version = ""
            elif "ppo" in model_type.lower():
                algo = "PPO"
                if "v3" in model_type or "v3" in str(checkpoint_dir):
                    version = "v3"
                elif "v2" in model_type or "v2" in str(checkpoint_dir):
                    version = "v2"
                else:
                    version = "v1"
            elif "grpo" in model_type.lower():
                algo = "GRPO"
                if "v3" in model_type or "v3" in str(checkpoint_dir):
                    version = "v3"
                elif "v2" in model_type or "v2" in str(checkpoint_dir):
                    version = "v2"
                else:
                    version = "v1"
            else:
                algo = "Unknown"
                version = ""
            
            model_name = f"{algo}" + (f"-{version}" if version else "")
            
            models.append({
                "name": model_name,
                "path": str(checkpoint_dir),
                "algorithm": algo,
                "version": version,
                "run": run_name
            })
    
    # Sort by algorithm and version
    models.sort(key=lambda x: (x["algorithm"], x["version"]))
    
    return models


def save_results_csv(results: Dict, output_path: Path):
    """Save results as CSV."""
    rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Top-1 Accuracy (%)": f"{metrics['accuracy'] * 100:.2f}",
            "Top-3 Accuracy (%)": f"{metrics['top3_accuracy'] * 100:.2f}",
            "Top-5 Accuracy (%)": f"{metrics['top5_accuracy'] * 100:.2f}",
            "Avg Selected Reward": f"{metrics['avg_selected_reward']:.4f}",
            "Avg Max Reward": f"{metrics['avg_max_reward']:.4f}",
            "Reward Ratio (%)": f"{metrics['reward_ratio'] * 100:.2f}",
            "Total Examples": metrics['total_examples']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV to {output_path}")


def save_results_latex(results: Dict, output_path: Path, baseline: str = None):
    """Save results as LaTeX table."""
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Comparison Results}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Model & Top-1 Acc & Top-3 Acc & Top-5 Acc & Selected Reward & Max Reward & Reward Ratio \\\\\n")
        f.write("\\hline\n")
        
        for model_name, metrics in results.items():
            acc1 = metrics['accuracy'] * 100
            acc3 = metrics['top3_accuracy'] * 100
            acc5 = metrics['top5_accuracy'] * 100
            sel_rew = metrics['avg_selected_reward']
            max_rew = metrics['avg_max_reward']
            rew_ratio = metrics['reward_ratio'] * 100
            
            # Highlight best values
            if baseline and baseline in results:
                baseline_acc = results[baseline]['accuracy'] * 100
                if acc1 > baseline_acc:
                    acc1_str = f"\\textbf{{{acc1:.2f}\\%}}"
                else:
                    acc1_str = f"{acc1:.2f}\\%"
            else:
                acc1_str = f"{acc1:.2f}\\%"
            
            f.write(f"{model_name} & {acc1_str} & {acc3:.2f}\\% & {acc5:.2f}\\% & {sel_rew:.4f} & {max_rew:.4f} & {rew_ratio:.2f}\\% \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logger.info(f"Saved LaTeX table to {output_path}")


def save_results_markdown(results: Dict, output_path: Path, baseline: str = None):
    """Save results as Markdown table."""
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Main results table
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Top-1 Acc | Top-3 Acc | Top-5 Acc | Selected Reward | Max Reward | Reward Ratio |\n")
        f.write("|-------|-----------|-----------|-----------|-----------------|------------|-------------|\n")
        
        for model_name, metrics in results.items():
            acc1 = f"{metrics['accuracy'] * 100:.2f}%"
            acc3 = f"{metrics['top3_accuracy'] * 100:.2f}%"
            acc5 = f"{metrics['top5_accuracy'] * 100:.2f}%"
            sel_rew = f"{metrics['avg_selected_reward']:.4f}"
            max_rew = f"{metrics['avg_max_reward']:.4f}"
            rew_ratio = f"{metrics['reward_ratio'] * 100:.2f}%"
            
            f.write(f"| {model_name} | {acc1} | {acc3} | {acc5} | {sel_rew} | {max_rew} | {rew_ratio} |\n")
        
        # Improvements section
        if baseline and baseline in results:
            f.write(f"\n## Improvements over {baseline}\n\n")
            f.write("| Model | Accuracy Δ | Reward Ratio Δ |\n")
            f.write("|-------|------------|----------------|\n")
            
            baseline_acc = results[baseline]['accuracy']
            baseline_ratio = results[baseline]['reward_ratio']
            
            for model_name, metrics in results.items():
                if model_name != baseline:
                    acc_diff = (metrics['accuracy'] - baseline_acc) * 100
                    ratio_diff = (metrics['reward_ratio'] - baseline_ratio) * 100
                    
                    acc_str = f"{acc_diff:+.2f}%"
                    ratio_str = f"{ratio_diff:+.2f}%"
                    
                    f.write(f"| {model_name} | {acc_str} | {ratio_str} |\n")
        
        # Summary statistics
        f.write("\n## Summary Statistics\n\n")
        f.write(f"- **Total Models Evaluated:** {len(results)}\n")
        f.write(f"- **Total Examples:** {list(results.values())[0]['total_examples']}\n")
        
        best_acc_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_reward_model = max(results.items(), key=lambda x: x[1]['reward_ratio'])
        
        f.write(f"- **Best Accuracy:** {best_acc_model[0]} ({best_acc_model[1]['accuracy']*100:.2f}%)\n")
        f.write(f"- **Best Reward Ratio:** {best_reward_model[0]} ({best_reward_model[1]['reward_ratio']*100:.2f}%)\n")
    
    logger.info(f"Saved Markdown table to {output_path}")


def create_comparison_plots(results: Dict, output_dir: Path):
    """Create comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(results.keys())
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Accuracy comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.25
    
    top1 = [results[m]['accuracy'] * 100 for m in models]
    top3 = [results[m]['top3_accuracy'] * 100 for m in models]
    top5 = [results[m]['top5_accuracy'] * 100 for m in models]
    
    bars1 = ax.bar(x - width, top1, width, label='Top-1', alpha=0.8)
    bars2 = ax.bar(x, top3, width, label='Top-3', alpha=0.8)
    bars3 = ax.bar(x + width, top5, width, label='Top-5', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Reward comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    selected = [results[m]['avg_selected_reward'] for m in models]
    maximum = [results[m]['avg_max_reward'] for m in models]
    
    x = np.arange(len(models))
    bars1 = ax.bar(x - width/2, selected, width, label='Selected Reward', alpha=0.8)
    bars2 = ax.bar(x + width/2, maximum, width, label='Max Possible', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Model Reward Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Reward ratio comparison (horizontal bar)
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.5 + 2))
    
    ratios = [results[m]['reward_ratio'] * 100 for m in models]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    bars = ax.barh(models, ratios, color=colors, alpha=0.8)
    
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(ratio + 0.2, i, f'{ratio:.2f}%', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Reward Ratio (%)', fontsize=12)
    ax.set_title('Reward Ratio by Model', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all experiment models")
    parser.add_argument("--base_dir", type=str, default="outputs",
                       help="Base directory containing model outputs")
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
    parser.add_argument("--test_split", type=str, default="val.jsonl",
                       help="Test split filename")
    parser.add_argument("--baseline", type=str, default="SFT",
                       help="Baseline model for comparison")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Find all models
    logger.info(f"Searching for models in {args.base_dir}...")
    model_list = find_all_models(args.base_dir)
    
    if not model_list:
        logger.error(f"No models found in {args.base_dir}")
        return
    
    logger.info(f"Found {len(model_list)} models:")
    for model_info in model_list:
        logger.info(f"  - {model_info['name']}: {model_info['path']}")
    
    # Load test data (use first model's tokenizer)
    logger.info("Loading test dataset...")
    _, tokenizer = load_model(model_list[0]['path'], args.base_model, device)
    
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
    for model_info in model_list:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_info['name']}")
        logger.info(f"{'='*80}")
        
        try:
            model, _ = load_model(model_info['path'], args.base_model, device)
            metrics = evaluate_model(model, test_loader, device)
            results[model_info['name']] = metrics
            
            logger.info(f"{model_info['name']} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
            logger.info(f"  Reward Ratio: {metrics['reward_ratio']:.2%}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model_info['name']}: {e}")
            continue
    
    if not results:
        logger.error("No models were successfully evaluated")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results in multiple formats
    logger.info("\n" + "="*80)
    logger.info("Saving results...")
    logger.info("="*80)
    
    # JSON (full results)
    results_json = {
        model_name: {k: v for k, v in metrics.items() 
                    if k not in ['all_selected_rewards', 'all_rewards']}
        for model_name, metrics in results.items()
    }
    results_json['metadata'] = {
        'timestamp': timestamp,
        'base_dir': args.base_dir,
        'test_split': args.test_split,
        'num_models': len(results)
    }
    
    with open(output_dir / f'results_{timestamp}.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved JSON to {output_dir / f'results_{timestamp}.json'}")
    
    # CSV
    save_results_csv(results, output_dir / f'results_{timestamp}.csv')
    
    # LaTeX
    save_results_latex(results, output_dir / f'results_{timestamp}.tex', 
                       baseline=args.baseline if args.baseline in results else None)
    
    # Markdown
    save_results_markdown(results, output_dir / f'results_{timestamp}.md',
                         baseline=args.baseline if args.baseline in results else None)
    
    # Plots
    create_comparison_plots(results, output_dir / f'plots_{timestamp}')
    
    # Also save as "latest" for easy access
    save_results_csv(results, output_dir / 'results_latest.csv')
    save_results_markdown(results, output_dir / 'results_latest.md',
                         baseline=args.baseline if args.baseline in results else None)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
