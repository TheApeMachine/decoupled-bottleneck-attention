#!/usr/bin/env python3
"""
analyze_run.py

Post-training analysis suite for understanding attention mechanisms.

Reads the JSONL logs and HDF5 tensor data from experiment runs and generates
comprehensive visualizations for understanding:
- Attention rank evolution
- Semantic vs Geometric path contributions  
- Gradient dynamics
- Head specialization

Usage:
    python3.12 analyze_run.py runs/paper_decoupled
    python3.12 analyze_run.py runs/paper_decoupled --compare runs/paper_baseline
    python3.12 analyze_run.py --all  # Analyze all runs in runs/paper_*
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import math

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class RunData:
    """Container for all data from an experiment run."""
    name: str
    path: Path
    config: Dict
    args: Dict
    
    # Basic metrics
    eval_steps: List[int]
    train_losses: List[float]
    val_losses: List[float]
    best_val: float
    total_time: float
    
    # Deep analysis (from instrumentation)
    attention_entropy: List[Dict]
    attention_rank: List[Dict]
    attention_sparsity: List[Dict]
    path_contribution: List[Dict]
    hidden_rank: List[Dict]
    gradient_norms: List[Dict]
    layer_similarity: List[Dict]


def load_run(run_dir: str) -> Optional[RunData]:
    """Load all data from an experiment run."""
    run_path = Path(run_dir)
    log_path = run_path / "train.jsonl"
    
    if not log_path.exists():
        print(f"  Warning: {log_path} not found")
        return None
    
    # Initialize containers
    config = {}
    args = {}
    eval_steps = []
    train_losses = []
    val_losses = []
    best_val = float('inf')
    total_time = 0
    
    attention_entropy = []
    attention_rank = []
    attention_sparsity = []
    path_contribution = []
    hidden_rank = []
    gradient_norms = []
    layer_similarity = []
    
    # Parse JSONL
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event_type = data.get("type")
                
                if event_type == "run_config":
                    config = data.get("config", {})
                    args = data.get("args", {})
                
                elif event_type == "eval":
                    eval_steps.append(data["step"])
                    train_losses.append(data["train_loss"])
                    val_losses.append(data["val_loss"])
                
                elif event_type == "best":
                    best_val = data["best_val"]
                
                elif event_type == "done":
                    best_val = data.get("best_val", best_val)
                    total_time = data.get("total_seconds", 0)
                
                elif event_type == "analysis":
                    step = data.get("step", 0)
                    
                    if "attention_entropy" in data:
                        attention_entropy.append({"step": step, **data["attention_entropy"]})
                    
                    if "attention_rank" in data:
                        attention_rank.append({"step": step, **data["attention_rank"]})
                    
                    if "attention_sparsity" in data:
                        attention_sparsity.append({"step": step, **data["attention_sparsity"]})
                    
                    if "path_contribution" in data:
                        path_contribution.append({"step": step, **data["path_contribution"]})
                    
                    if "hidden_rank" in data:
                        hidden_rank.append({"step": step, **data["hidden_rank"]})
                    
                    if "gradient_norms" in data:
                        gradient_norms.append({"step": step, **data["gradient_norms"]})
                    
                    if "layer_similarity" in data:
                        layer_similarity.append({"step": step, "values": data["layer_similarity"]})
                    
            except json.JSONDecodeError:
                continue
    
    return RunData(
        name=run_path.name,
        path=run_path,
        config=config,
        args=args,
        eval_steps=eval_steps,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val=min(val_losses) if val_losses else best_val,
        total_time=total_time,
        attention_entropy=attention_entropy,
        attention_rank=attention_rank,
        attention_sparsity=attention_sparsity,
        path_contribution=path_contribution,
        hidden_rank=hidden_rank,
        gradient_norms=gradient_norms,
        layer_similarity=layer_similarity,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_convergence(runs: List[RunData], output_path: Path):
    """Plot loss convergence for multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10.colors
    
    # Validation loss
    ax = axes[0]
    for i, run in enumerate(runs):
        if run.eval_steps and run.val_losses:
            ax.plot(run.eval_steps, run.val_losses, 
                   color=colors[i % len(colors)],
                   linewidth=2, marker='o', markersize=3,
                   label=f"{run.name} ({run.best_val:.4f})")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Train vs Val gap (overfitting indicator)
    ax = axes[1]
    for i, run in enumerate(runs):
        if run.eval_steps and run.train_losses and run.val_losses:
            gaps = [v - t for t, v in zip(run.train_losses, run.val_losses)]
            ax.plot(run.eval_steps, gaps,
                   color=colors[i % len(colors)],
                   linewidth=2, marker='o', markersize=3,
                   label=run.name)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Generalization Gap (higher = more overfit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {output_path}")


def plot_attention_analysis(run: RunData, output_path: Path):
    """Plot detailed attention analysis for a single run."""
    n_plots = sum([
        bool(run.attention_entropy),
        bool(run.attention_rank),
        bool(run.attention_sparsity),
        bool(run.path_contribution),
    ])
    
    if n_plots == 0:
        print(f"  No attention analysis data for {run.name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    plot_idx = 0
    
    # Attention Entropy
    if run.attention_entropy:
        ax = axes[plot_idx]
        steps = [d["step"] for d in run.attention_entropy]
        
        # Get all layer/head keys
        all_keys = set()
        for d in run.attention_entropy:
            all_keys.update(k for k in d.keys() if k != "step")
        
        for key in sorted(all_keys):
            vals = [d.get(key, None) for d in run.attention_entropy]
            vals = [v for v in vals if v is not None and v > 0]
            if vals:
                ax.plot(steps[:len(vals)], vals, alpha=0.5, linewidth=1)
        
        # Average
        avg_vals = []
        for d in run.attention_entropy:
            v = [val for k, val in d.items() if k != "step" and isinstance(val, (int, float)) and val > 0]
            avg_vals.append(sum(v) / len(v) if v else 0)
        ax.plot(steps, avg_vals, 'k-', linewidth=2.5, label='Average')
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.set_title("Attention Entropy Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Attention Rank
    if run.attention_rank:
        ax = axes[plot_idx]
        steps = [d["step"] for d in run.attention_rank]
        
        avg_vals = []
        for d in run.attention_rank:
            v = [val for k, val in d.items() if k != "step" and isinstance(val, (int, float)) and val > 0]
            avg_vals.append(sum(v) / len(v) if v else 0)
        
        ax.plot(steps, avg_vals, 'b-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective Rank")
        ax.set_title("Attention Effective Rank Over Training")
        ax.grid(True, alpha=0.3)
        
        # Add context dimension line if available
        if run.config.get("attn_dim"):
            ax.axhline(y=run.config["attn_dim"], color='r', linestyle='--', 
                      label=f'attn_dim={run.config["attn_dim"]}')
            ax.legend()
        
        plot_idx += 1
    
    # Attention Sparsity
    if run.attention_sparsity:
        ax = axes[plot_idx]
        steps = [d["step"] for d in run.attention_sparsity]
        
        avg_vals = []
        for d in run.attention_sparsity:
            v = [val for k, val in d.items() if k != "step" and isinstance(val, (int, float)) and val >= 0]
            avg_vals.append(sum(v) / len(v) if v else 0)
        
        ax.plot(steps, avg_vals, 'g-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Sparsity (fraction < 0.01)")
        ax.set_title("Attention Sparsity Over Training")
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Path Contribution (Decoupled attention)
    if run.path_contribution:
        ax = axes[plot_idx]
        steps = [d["step"] for d in run.path_contribution]
        
        sem_vals = []
        geo_vals = []
        for d in run.path_contribution:
            sem = []
            geo = []
            for k, v in d.items():
                if k != "step" and isinstance(v, dict):
                    if "semantic_ratio" in v and v["semantic_ratio"] >= 0:
                        sem.append(v["semantic_ratio"])
                    if "geometric_ratio" in v and v["geometric_ratio"] >= 0:
                        geo.append(v["geometric_ratio"])
            sem_vals.append(sum(sem) / len(sem) if sem else 0)
            geo_vals.append(sum(geo) / len(geo) if geo else 0)
        
        ax.plot(steps, sem_vals, 'b-', linewidth=2, label='Semantic')
        ax.plot(steps, geo_vals, 'r-', linewidth=2, label='Geometric')
        ax.set_xlabel("Step")
        ax.set_ylabel("Contribution Ratio")
        ax.set_title("Semantic vs Geometric Path Contribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"Attention Analysis: {run.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {output_path}")


def plot_gradient_analysis(run: RunData, output_path: Path):
    """Plot gradient analysis for a single run."""
    if not run.gradient_norms:
        print(f"  No gradient data for {run.name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = [d["step"] for d in run.gradient_norms]
    
    # Plot each component
    components = ["attn_grad_norm", "ffn_grad_norm", "embed_grad_norm", "other_grad_norm"]
    colors = ['blue', 'green', 'red', 'purple']
    
    for comp, color in zip(components, colors):
        vals = [d.get(comp, None) for d in run.gradient_norms]
        valid_vals = [(s, v) for s, v in zip(steps, vals) if v is not None and v > 0]
        if valid_vals:
            s, v = zip(*valid_vals)
            ax.plot(s, v, color=color, linewidth=1.5, 
                   label=comp.replace("_grad_norm", "").title())
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Gradient Norms: {run.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {output_path}")


def plot_rank_heatmap(run: RunData, output_path: Path):
    """Plot heatmap of attention rank per head over training."""
    if not run.attention_rank:
        print(f"  No rank data for {run.name}")
        return
    
    # Extract all layer/head keys
    all_keys = set()
    for d in run.attention_rank:
        all_keys.update(k for k in d.keys() if k != "step")
    
    if not all_keys:
        return
    
    keys = sorted(all_keys)
    steps = [d["step"] for d in run.attention_rank]
    
    # Build matrix
    matrix = np.zeros((len(keys), len(steps)))
    for j, d in enumerate(run.attention_rank):
        for i, k in enumerate(keys):
            val = d.get(k, 0)
            matrix[i, j] = val if isinstance(val, (int, float)) and val > 0 else 0
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(keys) * 0.3)))
    
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    
    # Downsample x ticks
    n_ticks = min(10, len(steps))
    tick_indices = np.linspace(0, len(steps)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([steps[i] for i in tick_indices])
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Layer/Head")
    ax.set_title(f"Attention Rank Evolution: {run.name}")
    
    plt.colorbar(im, ax=ax, label="Effective Rank")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {output_path}")


def generate_summary_report(runs: List[RunData], output_path: Path):
    """Generate a comprehensive markdown summary report."""
    lines = [
        "# Experiment Analysis Report",
        "",
        f"Analyzed {len(runs)} experiment runs.",
        "",
        "## Summary Table",
        "",
        "| Run | Best Val | PPL | Time (h) | Attn Mode | d_attn |",
        "|-----|----------|-----|----------|-----------|--------|",
    ]
    
    for run in sorted(runs, key=lambda r: r.best_val):
        ppl = math.exp(run.best_val) if run.best_val < 20 else float('inf')
        time_h = run.total_time / 3600 if run.total_time > 0 else 0
        attn_mode = run.config.get("attn_mode", "?")
        attn_dim = run.config.get("attn_dim", "?")
        lines.append(f"| {run.name} | {run.best_val:.4f} | {ppl:.1f} | {time_h:.2f} | {attn_mode} | {attn_dim} |")
    
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])
    
    # Find best and worst
    if runs:
        best = min(runs, key=lambda r: r.best_val)
        worst = max(runs, key=lambda r: r.best_val)
        
        lines.extend([
            f"- **Best model**: {best.name} (val={best.best_val:.4f})",
            f"- **Worst model**: {worst.name} (val={worst.best_val:.4f})",
            f"- **Gap**: {worst.best_val - best.best_val:.4f}",
            "",
        ])
    
    # Check for path contribution data (decoupled models)
    decoupled_runs = [r for r in runs if r.path_contribution]
    if decoupled_runs:
        lines.append("## Path Contribution Analysis (Decoupled Models)")
        lines.append("")
        
        for run in decoupled_runs:
            if run.path_contribution:
                final = run.path_contribution[-1]
                sem_vals = []
                geo_vals = []
                for k, v in final.items():
                    if k != "step" and isinstance(v, dict):
                        if "semantic_ratio" in v:
                            sem_vals.append(v["semantic_ratio"])
                        if "geometric_ratio" in v:
                            geo_vals.append(v["geometric_ratio"])
                
                if sem_vals:
                    sem_avg = sum(sem_vals) / len(sem_vals)
                    geo_avg = sum(geo_vals) / len(geo_vals)
                    lines.append(f"- **{run.name}**: Semantic={sem_avg*100:.1f}%, Geometric={geo_avg*100:.1f}%")
        
        lines.append("")
    
    # Check for rank evolution
    rank_runs = [r for r in runs if r.attention_rank]
    if rank_runs:
        lines.append("## Attention Rank Analysis")
        lines.append("")
        
        for run in rank_runs:
            if run.attention_rank:
                # Initial and final rank
                initial = run.attention_rank[0] if run.attention_rank else {}
                final = run.attention_rank[-1] if run.attention_rank else {}
                
                init_vals = [v for k, v in initial.items() if k != "step" and isinstance(v, (int, float)) and v > 0]
                final_vals = [v for k, v in final.items() if k != "step" and isinstance(v, (int, float)) and v > 0]
                
                if init_vals and final_vals:
                    init_avg = sum(init_vals) / len(init_vals)
                    final_avg = sum(final_vals) / len(final_vals)
                    lines.append(f"- **{run.name}**: Rank {init_avg:.1f} -> {final_avg:.1f} (Δ={final_avg - init_avg:+.1f})")
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  -> Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def analyze_single_run(run_dir: str):
    """Analyze a single run directory."""
    print(f"\nAnalyzing: {run_dir}")
    print("-" * 50)
    
    run = load_run(run_dir)
    if not run:
        print("  Failed to load run data")
        return
    
    print(f"  Config: {run.config.get('attn_mode', '?')} | d_attn={run.config.get('attn_dim', '?')}")
    print(f"  Best val: {run.best_val:.4f} | PPL: {math.exp(run.best_val):.1f}")
    print(f"  Analysis data: entropy={len(run.attention_entropy)}, rank={len(run.attention_rank)}, paths={len(run.path_contribution)}")
    
    if not HAS_MPL:
        print("  Warning: matplotlib not available, skipping plots")
        return
    
    output_dir = run.path / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_convergence([run], output_dir / "convergence.png")
    plot_attention_analysis(run, output_dir / "attention.png")
    plot_gradient_analysis(run, output_dir / "gradients.png")
    plot_rank_heatmap(run, output_dir / "rank_heatmap.png")


def analyze_all_paper_runs():
    """Analyze all runs in runs/ (paper_* or size-prefixed)"""
    runs_dir = Path("runs")
    
    # Find all experiment directories (paper_* or size_*)
    paper_dirs = sorted(runs_dir.glob("paper_*"))
    size_dirs = sorted(runs_dir.glob("tiny_*")) + sorted(runs_dir.glob("small_*")) + \
                sorted(runs_dir.glob("medium_*")) + sorted(runs_dir.glob("large_*"))
    
    all_dirs = sorted(set(paper_dirs + size_dirs))
    
    if not all_dirs:
        print("No experiment directories found (paper_* or {tiny,small,medium,large}_*)")
        return
    
    print(f"\nFound {len(all_dirs)} experiment directories")
    
    runs = []
    for run_dir in all_dirs:
        run = load_run(str(run_dir))
        if run:
            runs.append(run)
            print(f"  Loaded: {run.name} (val={run.best_val:.4f})")
    
    if not runs:
        print("No valid runs found")
        return
    
    if not HAS_MPL:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # Create comparison output directory
    output_dir = Path("assets/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_convergence(runs, output_dir / "comparison_convergence.png")
    generate_summary_report(runs, output_dir / "summary.md")
    
    # Individual analyses
    for run in runs:
        print(f"\nAnalyzing: {run.name}")
        run_output = output_dir / run.name
        run_output.mkdir(exist_ok=True)
        
        plot_attention_analysis(run, run_output / "attention.png")
        plot_gradient_analysis(run, run_output / "gradients.png")
        plot_rank_heatmap(run, run_output / "rank_heatmap.png")
    
    print(f"\nAnalysis complete! Results in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Post-training analysis suite")
    parser.add_argument("run_dir", nargs="?", help="Path to experiment run directory")
    parser.add_argument("--all", action="store_true", help="Analyze all runs/paper_* directories")
    parser.add_argument("--compare", nargs="+", help="Additional runs to compare")
    
    args = parser.parse_args()
    
    if args.all:
        analyze_all_paper_runs()
    elif args.run_dir:
        if args.compare:
            # Load all runs for comparison
            runs = []
            for run_dir in [args.run_dir] + args.compare:
                run = load_run(run_dir)
                if run:
                    runs.append(run)
            
            if runs:
                output_dir = Path("assets/comparison")
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_convergence(runs, output_dir / "convergence.png")
                generate_summary_report(runs, output_dir / "summary.md")
        else:
            analyze_single_run(args.run_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3.12 analyze_run.py runs/paper_decoupled")
        print("  python3.12 analyze_run.py --all")


if __name__ == "__main__":
    main()

