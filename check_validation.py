#!/usr/bin/env python3
"""
Check validation outputs vs ground truth.
Shows model predictions, ground truth, and reward scores.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def load_validation_samples(step: int = None):
    """Load validation samples from output directory."""
    valid_dir = Path("outputs/run_001/03_sft/valid_samples")
    
    if not valid_dir.exists():
        console.print("[red]❌ No validation samples found![/red]")
        console.print(f"Expected directory: {valid_dir}")
        return None
    
    # Find latest step if not specified
    if step is None:
        files = sorted(valid_dir.glob("step_*.json"))
        if not files:
            console.print("[red]❌ No validation files found![/red]")
            return None
        latest_file = files[-1]
    else:
        latest_file = valid_dir / f"step_{step}.json"
    
    if not latest_file.exists():
        console.print(f"[red]❌ File not found: {latest_file}[/red]")
        return None
    
    console.print(f"[blue]📄 Loading: {latest_file}[/blue]\n")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def display_sample(sample: dict, idx: int):
    """Display a single validation sample with rich formatting."""
    
    # Header
    console.print(Panel(
        f"Sample {idx + 1} | Reward: {sample.get('reward', 0):.3f}",
        style="bold cyan"
    ))
    
    # Question (truncated)
    question = sample.get('question', '')
    if len(question) > 100:
        question = question[:100] + "..."
    console.print(f"[yellow]Question:[/yellow] {question}\n")
    
    # Ground Truth
    console.print("[green]Ground Truth:[/green]")
    try:
        gt = json.loads(sample.get('ground_truth', '{}'))
        console.print(Syntax(json.dumps(gt, indent=2), "json", theme="monokai"))
    except:
        console.print(sample.get('ground_truth', 'N/A'))
    console.print()
    
    # Prediction
    console.print("[blue]Prediction:[/blue]")
    try:
        pred = json.loads(sample.get('prediction', '{}'))
        console.print(Syntax(json.dumps(pred, indent=2), "json", theme="monokai"))
    except:
        console.print(sample.get('prediction', 'N/A'))
    
    console.print("\n" + "="*80 + "\n")


def show_summary(samples: list):
    """Show summary statistics."""
    total = len(samples)
    total_reward = sum(s.get('reward', 0) for s in samples)
    avg_reward = total_reward / total if total > 0 else 0
    
    # Count parse successes
    parse_success = sum(1 for s in samples if is_valid_json(s.get('prediction', '')))
    parse_rate = (parse_success / total * 100) if total > 0 else 0
    
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Samples", str(total))
    table.add_row("Avg Reward", f"{avg_reward:.4f}")
    table.add_row("Parse Rate", f"{parse_rate:.1f}%")
    table.add_row("Parsed Samples", f"{parse_success}/{total}")
    
    console.print(table)
    console.print()


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except:
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check validation outputs")
    parser.add_argument("--step", type=int, help="Validation step to check (default: latest)")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to show (default: 5)")
    parser.add_argument("--all", action="store_true", help="Show all samples")
    args = parser.parse_args()
    
    # Load samples
    samples = load_validation_samples(args.step)
    if samples is None:
        return 1
    
    # Show summary
    show_summary(samples)
    
    # Show individual samples
    num_to_show = len(samples) if args.all else min(args.num, len(samples))
    console.print(f"[bold]Showing {num_to_show} samples:[/bold]\n")
    
    for i in range(num_to_show):
        display_sample(samples[i], i)
    
    if not args.all and len(samples) > num_to_show:
        console.print(f"[dim]... and {len(samples) - num_to_show} more samples[/dim]")
        console.print(f"[dim]Use --all to see all samples or --num N to see N samples[/dim]")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        exit(1)
