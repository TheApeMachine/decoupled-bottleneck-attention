#!/usr/bin/env python3
"""
plot_results.py
Generate convergence plots, Pareto curves, and comparison visualizations.

Usage:
    python3 plot_results.py

Generates:
    - assets/convergence_wikitext.png   (WikiText-2 experiments)
    - assets/convergence_fineweb.png    (FineWeb-Edu experiments)
    - assets/pareto_curve.png           (All experiments on one Pareto plot)
    - assets/comparison_bar.png         (Bar chart comparing final losses)
"""
import matplotlib.pyplot as plt
import re
import os
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# WikiText-2 Experiments
# Format: "label": ("path", "format") where format is "log" or "jsonl"
WIKITEXT_LOGS = {
    "Standard 512 (v19)": ("runs/v19_baseline/train.jsonl", "jsonl"),
    "⭐ Combined 96": ("runs/v21_combined_baseline_96/train.log", "log"),
    "GQA (kv=2)": ("runs/v21_gqa_kv2_parammatch/train.log", "log"),
    "Small Model (d=128)": ("runs/v21_small_d128_standard/train.log", "log"),
    # Note: The below are SHORT RUNS (not 6000 steps) - shown for reference only
    "Decoupled 1024 ctx*": ("runs/v21_decoupled_sem32_geo64_block1024/train.log", "log"),
    "Decoupled 2048 ctx*": ("runs/v21_decoupled_sem32_geo64_block2048/train.log", "log"),
}

# FineWeb-Edu Experiments (100M tokens, 1024 context)
FINEWEB_LOGS = {
    "Baseline (512)": ("runs/v21_fineweb_baseline/train.log", "log"),
    "Decoupled (32/64)": ("runs/v21_fineweb_decoupled/train.log", "log"),
}

# Pareto Data: (attn_dim, final_val_loss, label, dataset)
# Values verified from actual experiment logs
PARETO_DATA = [
    # WikiText-2 (verified)
    (512, 5.3687, "Standard 512 (WT2)", "wikitext"),      # v19_baseline
    (96, 5.3272, "⭐ Combined 96 (WT2)", "wikitext"),     # v21_combined_baseline_96
    # (128, 5.48, "Bottleneck 128 (WT2)", "wikitext"),   # NOT YET RUN
    # (96, 5.59, "Decoupled 32/64 (WT2)", "wikitext"),   # NEEDS RE-RUN (was short/wrong params)
    (128, 5.6320, "GQA kv=2 (WT2)", "wikitext"),         # v21_gqa_kv2_parammatch
    (128, 5.7428, "Small d=128 (WT2)", "wikitext"),      # v21_small_d128_standard
    # FineWeb-Edu (verified)
    (512, 4.0989, "Baseline 512 (FW)", "fineweb"),       # v21_fineweb_baseline
    (96, 4.4915, "Decoupled 32/64 (FW)", "fineweb"),     # v21_fineweb_decoupled
]

# Output paths
OUTPUT_DIR = Path("assets")

# Styling
COLORS = {
    "wikitext": "#2196F3",   # Blue for WikiText
    "fineweb": "#FF5722",    # Orange for FineWeb
    "highlight": "#4CAF50",  # Green for highlights
}


def parse_log(filepath: str, fmt: str = "log") -> tuple[list[int], list[float]]:
    """
    Parse training log to extract eval steps and validation losses.
    
    Args:
        filepath: Path to log file
        fmt: "log" for text format, "jsonl" for v19-style JSONL format
    
    Expected log format:
        == eval step 200 | train 5.3370 | val 5.8770 | val_ppl 356.74 | 203.3s
    
    Expected JSONL format:
        {"type": "eval", "step": 200, "val_loss": 5.8770, ...}
    """
    steps = []
    val_losses = []
    
    if not os.path.exists(filepath):
        print(f"  ⚠ Warning: {filepath} not found")
        return [], []
    
    with open(filepath, 'r') as f:
        if fmt == "jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "eval" and "step" in data and "val_loss" in data:
                        steps.append(int(data["step"]))
                        val_losses.append(float(data["val_loss"]))
                except json.JSONDecodeError:
                    continue
        else:
            for line in f:
                match = re.search(r"== eval step (\d+) .* val (\d+\.\d+)", line)
                if match:
                    steps.append(int(match.group(1)))
                    val_losses.append(float(match.group(2)))
    
    return steps, val_losses


def plot_convergence_wikitext():
    """Generate convergence plot for WikiText-2 experiments."""
    plt.figure(figsize=(12, 7))
    
    found_any = False
    for label, (path, fmt) in WIKITEXT_LOGS.items():
        steps, losses = parse_log(path, fmt)
        if steps:
            # Highlight baseline and best model
            if "Standard" in label:
                plt.plot(steps, losses, label=label, marker='s', markersize=4, 
                        linewidth=2.5, linestyle='--', color='#333333')
            elif '⭐' in label:
                plt.plot(steps, losses, label=label, marker='o', markersize=4, 
                        linewidth=2.5, color='#4CAF50')
            else:
                plt.plot(steps, losses, label=label, marker='o', markersize=3, linewidth=1.5)
            found_any = True
            print(f"  ✓ {label}: {len(steps)} points, best = {min(losses):.4f}")
    
    if not found_any:
        print("  No WikiText logs found!")
        return
    
    plt.title("WikiText-2: Validation Loss Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss (Lower is Better)", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / "convergence_wikitext.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  → Saved {output_path}")
    plt.close()


def plot_convergence_fineweb():
    """Generate convergence plot for FineWeb-Edu experiments."""
    plt.figure(figsize=(12, 7))
    
    found_any = False
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']  # Blue, Orange, Green, Purple
    
    for i, (label, (path, fmt)) in enumerate(FINEWEB_LOGS.items()):
        steps, losses = parse_log(path, fmt)
        if steps:
            plt.plot(steps, losses, label=label, marker='o', markersize=4, 
                     linewidth=2.5, color=colors[i % len(colors)])
            found_any = True
            print(f"  ✓ {label}: {len(steps)} points, best = {min(losses):.4f}")
    
    if not found_any:
        print("  No FineWeb logs found!")
        return
    
    plt.title("FineWeb-Edu (100M tokens): Validation Loss Convergence", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss (Lower is Better)", fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add annotation for the gap
    plt.annotate(
        f"Gap: ~0.39 loss\n(9.6% relative)",
        xy=(6000, 4.3), fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    output_path = OUTPUT_DIR / "convergence_fineweb.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  → Saved {output_path}")
    plt.close()


def plot_combined_convergence():
    """Generate a combined convergence plot with all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # WikiText-2 (left panel)
    for label, (path, fmt) in WIKITEXT_LOGS.items():
        steps, losses = parse_log(path, fmt)
        if steps:
            ax1.plot(steps, losses, label=label, marker='o', markersize=3, linewidth=2)
    
    ax1.set_title("WikiText-2 (256 context)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Validation Loss", fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # FineWeb-Edu (right panel)
    colors_fw = ['#2196F3', '#FF5722']
    for i, (label, (path, fmt)) in enumerate(FINEWEB_LOGS.items()):
        steps, losses = parse_log(path, fmt)
        if steps:
            ax2.plot(steps, losses, label=label, marker='o', markersize=4, 
                     linewidth=2.5, color=colors_fw[i])
    
    ax2.set_title("FineWeb-Edu (1024 context, 100M tokens)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "convergence_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  → Saved {output_path}")
    plt.close()


def plot_pareto():
    """Generate Pareto curve with both WikiText and FineWeb data."""
    plt.figure(figsize=(12, 7))
    
    # Separate by dataset
    wt_data = [(d[0], d[1], d[2]) for d in PARETO_DATA if d[3] == "wikitext"]
    fw_data = [(d[0], d[1], d[2]) for d in PARETO_DATA if d[3] == "fineweb"]
    
    # Plot WikiText-2 points
    wt_dims = [d[0] for d in wt_data]
    wt_losses = [d[1] for d in wt_data]
    wt_labels = [d[2] for d in wt_data]
    
    plt.scatter(wt_dims, wt_losses, c=COLORS["wikitext"], s=120, zorder=5, 
                label="WikiText-2", edgecolors='white', linewidth=1.5)
    
    for i, label in enumerate(wt_labels):
        offset = (8, 5) if "⭐" not in label else (8, -12)
        plt.annotate(label.replace(" (WT2)", ""), (wt_dims[i], wt_losses[i]), 
                     textcoords="offset points", xytext=offset, 
                     ha='left', fontsize=8, color=COLORS["wikitext"])
    
    # Plot FineWeb points
    fw_dims = [d[0] for d in fw_data]
    fw_losses = [d[1] for d in fw_data]
    fw_labels = [d[2] for d in fw_data]
    
    plt.scatter(fw_dims, fw_losses, c=COLORS["fineweb"], s=120, zorder=5, 
                label="FineWeb-Edu", marker='s', edgecolors='white', linewidth=1.5)
    
    for i, label in enumerate(fw_labels):
        plt.annotate(label.replace(" (FW)", ""), (fw_dims[i], fw_losses[i]), 
                     textcoords="offset points", xytext=(8, 5), 
                     ha='left', fontsize=8, color=COLORS["fineweb"])
    
    # Highlight the tradeoff zone
    plt.axhspan(4.0, 4.6, alpha=0.1, color='orange', label='FineWeb Zone')
    plt.axhspan(5.3, 5.8, alpha=0.1, color='blue', label='WikiText Zone')
    
    plt.title("Pareto Frontier: Attention Dimension vs. Perplexity", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Attention Dimension (d_attn)", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks([64, 96, 128, 256, 512], ['64', '96', '128', '256', '512'])
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / "pareto_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  → Saved {output_path}")
    plt.close()


def plot_comparison_bar():
    """Bar chart comparing final validation losses across all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WikiText-2 comparison
    wt_labels = []
    wt_values = []
    for label, (path, fmt) in WIKITEXT_LOGS.items():
        steps, losses = parse_log(path, fmt)
        if steps:
            wt_labels.append(label.replace("⭐ ", ""))
            wt_values.append(min(losses))
    
    if wt_values:
        colors = [COLORS["highlight"] if "Combined" in l else COLORS["wikitext"] 
                  for l in wt_labels]
        bars1 = ax1.barh(wt_labels, wt_values, color=colors, edgecolor='white')
        ax1.set_xlabel("Best Validation Loss", fontsize=11)
        ax1.set_title("WikiText-2 Results", fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        for bar, val in zip(bars1, wt_values):
            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                     f'{val:.3f}', va='center', fontsize=9)
    
    # FineWeb comparison
    fw_labels = []
    fw_values = []
    for label, (path, fmt) in FINEWEB_LOGS.items():
        steps, losses = parse_log(path, fmt)
        if steps:
            fw_labels.append(label)
            fw_values.append(min(losses))
    
    if fw_values:
        colors = [COLORS["fineweb"] if "Baseline" in l else '#FF8A65' for l in fw_labels]
        bars2 = ax2.barh(fw_labels, fw_values, color=colors, edgecolor='white')
        ax2.set_xlabel("Best Validation Loss", fontsize=11)
        ax2.set_title("FineWeb-Edu Results", fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        for bar, val in zip(bars2, fw_values):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                     f'{val:.3f}', va='center', fontsize=9)
        
        # Add gap annotation
        if len(fw_values) >= 2:
            gap = fw_values[1] - fw_values[0]  # Decoupled - Baseline
            gap_pct = (gap / fw_values[0]) * 100
            ax2.annotate(f"Δ = {gap:.3f} ({gap_pct:.1f}%)", 
                         xy=(max(fw_values) * 0.5, 0.5), fontsize=11,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "comparison_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  → Saved {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("  Generating Research Visualizations")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. WikiText-2 Convergence Plot")
    print("-" * 50)
    plot_convergence_wikitext()
    
    print("\n2. FineWeb-Edu Convergence Plot")
    print("-" * 50)
    plot_convergence_fineweb()
    
    print("\n3. Combined Convergence (Side-by-Side)")
    print("-" * 50)
    plot_combined_convergence()
    
    print("\n4. Pareto Curve (All Experiments)")
    print("-" * 50)
    plot_pareto()
    
    print("\n5. Comparison Bar Chart")
    print("-" * 50)
    plot_comparison_bar()
    
    print("\n" + "=" * 70)
    print("  Done! Check assets/ folder for all visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
