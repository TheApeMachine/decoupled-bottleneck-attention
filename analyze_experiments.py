#!/usr/bin/env python3
"""
analyze_experiments.py
Parse experiment logs and generate analysis tables/plots.

Usage:
    python3 analyze_experiments.py --mode ablation      # (d_sem, d_geo) ablation table
    python3 analyze_experiments.py --mode multiseed    # Confidence intervals
    python3 analyze_experiments.py --mode all          # Both analyses

Generates:
    - assets/ablation_table.png       (d_sem, d_geo ablation results)
    - assets/multiseed_table.png      (Multi-seed confidence intervals)
    - assets/ablation_data.csv        (Raw ablation data)
    - assets/multiseed_data.csv       (Raw multi-seed data)
"""
import argparse
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import math

# Try importing matplotlib (optional for table generation)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Tables will be text-only.")


@dataclass
class ExperimentResult:
    name: str
    best_val_loss: float
    final_val_loss: float
    best_val_ppl: float
    steps: int
    seed: Optional[int] = None
    sem_dim: Optional[int] = None
    geo_dim: Optional[int] = None


def parse_log(filepath: str) -> Optional[ExperimentResult]:
    """Parse a training log file and extract key metrics."""
    if not os.path.exists(filepath):
        return None
    
    val_losses = []
    val_ppls = []
    steps = []
    best_val = float('inf')
    
    with open(filepath, 'r') as f:
        for line in f:
            # Parse eval lines: == eval step 200 | train 5.3370 | val 5.8770 | val_ppl 356.74 | 203.3s
            match = re.search(r"== eval step (\d+) .* val (\d+\.\d+) \| val_ppl (\d+\.\d+)", line)
            if match:
                step = int(match.group(1))
                val_loss = float(match.group(2))
                val_ppl = float(match.group(3))
                steps.append(step)
                val_losses.append(val_loss)
                val_ppls.append(val_ppl)
            
            # Parse best_val from end of log
            best_match = re.search(r"best_val=(\d+\.\d+)", line)
            if best_match:
                best_val = float(best_match.group(1))
    
    if not val_losses:
        return None
    
    return ExperimentResult(
        name=Path(filepath).parent.name,
        best_val_loss=best_val if best_val != float('inf') else min(val_losses),
        final_val_loss=val_losses[-1],
        best_val_ppl=math.exp(min(val_losses)),
        steps=steps[-1] if steps else 0
    )


def analyze_ablation():
    """Analyze (d_sem, d_geo) ablation experiments."""
    print("\n" + "=" * 70)
    print("  (d_sem, d_geo) Ablation Analysis")
    print("=" * 70)
    
    # Define ablation experiments
    ablation_configs = [
        ("16/80", 16, 80, "runs/ablation_sem16_geo80/train.log"),
        ("24/72", 24, 72, "runs/ablation_sem24_geo72/train.log"),
        ("32/64", 32, 64, "runs/ablation_sem32_geo64/train.log"),
        ("48/48", 48, 48, "runs/ablation_sem48_geo48/train.log"),
        ("64/32", 64, 32, "runs/ablation_sem64_geo32/train.log"),
    ]
    
    results = []
    for name, sem, geo, path in ablation_configs:
        result = parse_log(path)
        if result:
            result.sem_dim = sem
            result.geo_dim = geo
            results.append(result)
            print(f"  ✓ {name}: val_loss = {result.best_val_loss:.4f}")
        else:
            print(f"  ⚠ {name}: not found ({path})")
    
    if not results:
        print("\n  No ablation results found. Run 'make ablation_sem_geo' first.")
        return
    
    # Sort by best_val_loss
    results.sort(key=lambda x: x.best_val_loss)
    
    # Print table
    print("\n" + "-" * 70)
    print(f"  {'Config':<10} {'d_sem':<8} {'d_geo':<8} {'Val Loss':<12} {'Δ Best':<10}")
    print("-" * 70)
    
    best_loss = results[0].best_val_loss
    for r in results:
        delta = ((r.best_val_loss - best_loss) / best_loss) * 100
        marker = "⭐" if r == results[0] else "  "
        print(f"{marker} {r.sem_dim}/{r.geo_dim:<7} {r.sem_dim:<8} {r.geo_dim:<8} {r.best_val_loss:<12.4f} {delta:+.2f}%")
    
    print("-" * 70)
    
    # Save to CSV
    csv_path = Path("assets/ablation_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("config,sem_dim,geo_dim,best_val_loss,delta_pct\n")
        for r in results:
            delta = ((r.best_val_loss - best_loss) / best_loss) * 100
            f.write(f"{r.sem_dim}/{r.geo_dim},{r.sem_dim},{r.geo_dim},{r.best_val_loss:.4f},{delta:.2f}\n")
    print(f"\n  Saved: {csv_path}")
    
    # Generate plot if matplotlib available
    if HAS_MATPLOTLIB and len(results) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot as bar chart
        configs = [f"{r.sem_dim}/{r.geo_dim}" for r in results]
        losses = [r.best_val_loss for r in results]
        colors = ['#4CAF50' if r == results[0] else '#2196F3' for r in results]
        
        bars = ax.bar(configs, losses, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('(d_sem / d_geo) Configuration', fontsize=12)
        ax.set_ylabel('Best Validation Loss', fontsize=12)
        ax.set_title('Semantic/Geometric Dimension Ablation (WikiText-2)', fontsize=14, fontweight='bold')
        ax.set_ylim(min(losses) * 0.98, max(losses) * 1.02)
        
        # Add legend
        best_patch = mpatches.Patch(color='#4CAF50', label='Best Configuration')
        ax.legend(handles=[best_patch], loc='upper right')
        
        plt.tight_layout()
        plot_path = Path("assets/ablation_table.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    
    return results


def analyze_multiseed():
    """Analyze multi-seed validation experiments and compute confidence intervals."""
    print("\n" + "=" * 70)
    print("  Multi-Seed Validation Analysis")
    print("=" * 70)
    
    # Define multi-seed experiment groups
    seed_groups = {
        "Baseline (512)": [
            ("s1337", "runs/multiseed_baseline_s1337/train.log"),
            ("s42", "runs/multiseed_baseline_s42/train.log"),
            ("s123", "runs/multiseed_baseline_s123/train.log"),
        ],
        "Combined 96": [
            ("s1337", "runs/multiseed_combined96_s1337/train.log"),
            ("s42", "runs/multiseed_combined96_s42/train.log"),
            ("s123", "runs/multiseed_combined96_s123/train.log"),
        ],
        "Decoupled 32/64": [
            ("s1337", "runs/multiseed_decoupled_s1337/train.log"),
            ("s42", "runs/multiseed_decoupled_s42/train.log"),
            ("s123", "runs/multiseed_decoupled_s123/train.log"),
        ],
    }
    
    summary = []
    
    for group_name, seed_configs in seed_groups.items():
        losses = []
        for seed_name, path in seed_configs:
            result = parse_log(path)
            if result:
                losses.append(result.best_val_loss)
                print(f"  ✓ {group_name} ({seed_name}): {result.best_val_loss:.4f}")
            else:
                print(f"  ⚠ {group_name} ({seed_name}): not found")
        
        if len(losses) >= 2:
            mean = sum(losses) / len(losses)
            variance = sum((x - mean) ** 2 for x in losses) / len(losses)
            std = math.sqrt(variance)
            # 95% CI for small samples (t-distribution approximation)
            ci_95 = 2.0 * std / math.sqrt(len(losses))  # Simplified for n=3
            summary.append({
                "name": group_name,
                "mean": mean,
                "std": std,
                "ci_95": ci_95,
                "n": len(losses),
                "losses": losses
            })
    
    if not summary:
        print("\n  No multi-seed results found. Run 'make multiseed_validation' first.")
        return
    
    # Print summary table
    print("\n" + "-" * 70)
    print(f"  {'Model':<20} {'Mean':<10} {'Std':<10} {'95% CI':<15} {'n':<5}")
    print("-" * 70)
    
    for s in summary:
        ci_str = f"±{s['ci_95']:.3f}"
        print(f"  {s['name']:<20} {s['mean']:<10.4f} {s['std']:<10.4f} {ci_str:<15} {s['n']:<5}")
    
    print("-" * 70)
    
    # Statistical comparison: Is Combined 96 significantly better than Baseline?
    if len(summary) >= 2:
        baseline = next((s for s in summary if "Baseline" in s["name"]), None)
        combined = next((s for s in summary if "Combined" in s["name"]), None)
        
        if baseline and combined:
            diff = baseline["mean"] - combined["mean"]
            combined_std = math.sqrt(baseline["std"]**2 + combined["std"]**2)
            
            print(f"\n  Baseline - Combined 96: {diff:+.4f} (pooled std: {combined_std:.4f})")
            if diff > 0:
                print(f"  → Combined 96 is better by {diff:.4f} loss ({(diff/baseline['mean'])*100:.1f}%)")
            
            # Check if CIs overlap
            baseline_low = baseline["mean"] - baseline["ci_95"]
            baseline_high = baseline["mean"] + baseline["ci_95"]
            combined_low = combined["mean"] - combined["ci_95"]
            combined_high = combined["mean"] + combined["ci_95"]
            
            if combined_high < baseline_low:
                print("  → 95% confidence intervals DO NOT overlap (statistically significant)")
            else:
                print("  → 95% confidence intervals OVERLAP (not statistically significant)")
    
    # Save to CSV
    csv_path = Path("assets/multiseed_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("model,mean,std,ci_95,n,seed_losses\n")
        for s in summary:
            losses_str = ";".join(f"{l:.4f}" for l in s["losses"])
            f.write(f"{s['name']},{s['mean']:.4f},{s['std']:.4f},{s['ci_95']:.4f},{s['n']},{losses_str}\n")
    print(f"\n  Saved: {csv_path}")
    
    # Generate plot if matplotlib available
    if HAS_MATPLOTLIB and summary:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [s["name"] for s in summary]
        means = [s["mean"] for s in summary]
        errors = [s["ci_95"] for s in summary]
        colors = ['#4CAF50' if "Combined" in n else '#2196F3' for n in names]
        
        bars = ax.bar(names, means, yerr=errors, color=colors, 
                      edgecolor='white', linewidth=2, capsize=8, error_kw={'linewidth': 2})
        
        # Add value labels
        for bar, mean, ci in zip(bars, means, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.02,
                    f'{mean:.3f}±{ci:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Best Validation Loss', fontsize=12)
        ax.set_title('Multi-Seed Validation (WikiText-2, n=3)', fontsize=14, fontweight='bold')
        ax.set_ylim(min(means) * 0.95, max(means) * 1.08)
        
        plt.tight_layout()
        plot_path = Path("assets/multiseed_table.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    
    return summary


def analyze_long_context():
    """Analyze context length scaling experiments."""
    print("\n" + "=" * 70)
    print("  Long Context Scaling Analysis")
    print("=" * 70)
    
    # Define context scaling experiments
    context_configs = [
        (256, "runs/context_256/train.log"),
        (512, "runs/context_512/train.log"),
        (1024, "runs/context_1024/train.log"),
        (2048, "runs/context_2048/train.log"),
        (4096, "runs/context_4096/train.log"),
    ]
    
    baseline_configs = [
        (1024, "runs/baseline_context_1024/train.log"),
        (2048, "runs/baseline_context_2048/train.log"),
        (4096, "runs/baseline_context_4096/train.log"),
    ]
    
    decoupled_results = []
    baseline_results = []
    
    print("\nDecoupled (32/64) results:")
    for ctx, path in context_configs:
        result = parse_log(path)
        if result:
            decoupled_results.append((ctx, result.best_val_loss))
            print(f"  ✓ Context {ctx}: val_loss = {result.best_val_loss:.4f}")
        else:
            print(f"  ⚠ Context {ctx}: not found ({path})")
    
    print("\nBaseline (512) results:")
    for ctx, path in baseline_configs:
        result = parse_log(path)
        if result:
            baseline_results.append((ctx, result.best_val_loss))
            print(f"  ✓ Context {ctx}: val_loss = {result.best_val_loss:.4f}")
        else:
            print(f"  ⚠ Context {ctx}: not found ({path})")
    
    if not decoupled_results:
        print("\n  No long context results found. Run 'make long_context_scaling' first.")
        return
    
    # Print comparison table
    print("\n" + "-" * 70)
    print(f"  {'Context':<12} {'Decoupled':<15} {'Baseline':<15} {'Gap':<10}")
    print("-" * 70)
    
    for ctx, dec_loss in decoupled_results:
        baseline_loss = next((bl for c, bl in baseline_results if c == ctx), None)
        if baseline_loss:
            gap = ((dec_loss - baseline_loss) / baseline_loss) * 100
            gap_str = f"+{gap:.1f}%"
        else:
            gap_str = "N/A"
            baseline_loss = "—"
        
        print(f"  {ctx:<12} {dec_loss:<15.4f} {str(baseline_loss):<15} {gap_str:<10}")
    
    print("-" * 70)
    
    # Analyze scaling behavior
    if len(decoupled_results) >= 2:
        # Calculate how loss scales with context
        first_ctx, first_loss = decoupled_results[0]
        last_ctx, last_loss = decoupled_results[-1]
        
        ctx_ratio = last_ctx / first_ctx
        loss_increase = last_loss - first_loss
        
        print(f"\nScaling Analysis:")
        print(f"  Context increase: {first_ctx} → {last_ctx} ({ctx_ratio:.0f}x)")
        print(f"  Loss increase: {first_loss:.4f} → {last_loss:.4f} (+{loss_increase:.4f})")
        print(f"  Loss per 2x context: ~{loss_increase / math.log2(ctx_ratio):.4f}")
        
        # Extrapolation to 128k
        if last_ctx < 128000:
            extrap_factor = math.log2(128000 / last_ctx)
            projected_loss = last_loss + (loss_increase / math.log2(ctx_ratio)) * extrap_factor
            print(f"\n  Projected loss at 128k context: ~{projected_loss:.2f}")
            print(f"  (Linear extrapolation - actual may differ)")
    
    # Save to CSV
    csv_path = Path("assets/long_context_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("context,decoupled_loss,baseline_loss,gap_pct\n")
        for ctx, dec_loss in decoupled_results:
            baseline_loss = next((bl for c, bl in baseline_results if c == ctx), None)
            gap = ((dec_loss - baseline_loss) / baseline_loss) * 100 if baseline_loss else 0
            bl_str = f"{baseline_loss:.4f}" if baseline_loss else ""
            f.write(f"{ctx},{dec_loss:.4f},{bl_str},{gap:.2f}\n")
    print(f"\n  Saved: {csv_path}")
    
    # Generate plot if matplotlib available
    if HAS_MATPLOTLIB and decoupled_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot decoupled
        dec_ctx = [r[0] for r in decoupled_results]
        dec_loss = [r[1] for r in decoupled_results]
        ax.plot(dec_ctx, dec_loss, 'o-', linewidth=2, markersize=8, 
                label='Decoupled (32/64)', color='#FF5722')
        
        # Plot baseline
        if baseline_results:
            base_ctx = [r[0] for r in baseline_results]
            base_loss = [r[1] for r in baseline_results]
            ax.plot(base_ctx, base_loss, 's-', linewidth=2, markersize=8,
                    label='Baseline (512)', color='#2196F3')
        
        ax.set_xlabel('Context Length', fontsize=12)
        ax.set_ylabel('Best Validation Loss', fontsize=12)
        ax.set_title('Perplexity Scaling with Context Length (FineWeb-Edu)', 
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add extrapolation line to 128k
        if len(decoupled_results) >= 2:
            # Fit log-linear trend
            log_ctx = [math.log2(c) for c in dec_ctx]
            # Simple linear regression
            n = len(log_ctx)
            sum_x = sum(log_ctx)
            sum_y = sum(dec_loss)
            sum_xy = sum(x*y for x, y in zip(log_ctx, dec_loss))
            sum_x2 = sum(x*x for x in log_ctx)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Project to 128k
            extrap_ctx = [max(dec_ctx), 8192, 16384, 32768, 65536, 131072]
            extrap_loss = [slope * math.log2(c) + intercept for c in extrap_ctx]
            
            ax.plot(extrap_ctx, extrap_loss, '--', color='#FF5722', alpha=0.5,
                    label='Projected (Decoupled)')
            ax.axvline(x=131072, color='gray', linestyle=':', alpha=0.5)
            ax.text(131072, max(dec_loss), '128k', rotation=90, va='bottom')
        
        plt.tight_layout()
        plot_path = Path("assets/context_scaling.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close()
    
    return decoupled_results, baseline_results


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['ablation', 'multiseed', 'long_context', 'all'],
                        help='Analysis mode')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Experiment Analysis")
    print("=" * 70)
    
    if args.mode in ('ablation', 'all'):
        analyze_ablation()
    
    if args.mode in ('multiseed', 'all'):
        analyze_multiseed()
    
    if args.mode in ('long_context', 'all'):
        analyze_long_context()
    
    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

