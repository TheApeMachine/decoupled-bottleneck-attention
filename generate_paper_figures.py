#!/usr/bin/env python3
"""
generate_paper_figures.py

Generates all figures for the paper by reading instrumented experiment logs.
This script reads the JSONL logs from runs/paper_*/ directories and creates
publication-ready visualizations.

Usage:
    python3.12 generate_paper_figures.py

Outputs:
    assets/fig1_convergence_wikitext.png    - WikiText-2 convergence comparison
    assets/fig2_convergence_fineweb.png     - FineWeb-Edu convergence comparison
    assets/fig3_pareto.png                  - Memory vs Quality trade-off
    assets/fig4_comparison_bar.png          - Final loss comparison
    assets/table1_results.tex               - LaTeX table for WikiText-2
    assets/table2_results.tex               - LaTeX table for FineWeb
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment directories to parse (FineWeb-Edu only)
# Will auto-detect SIZE from run directory names (tiny_, small_, medium_, large_)

def discover_experiments() -> dict:
    """Auto-discover experiment runs by scanning runs/ directory."""
    import os
    from pathlib import Path
    
    experiments = {}
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        return experiments
    
    # Look for size-prefixed directories
    for size in ["tiny", "small", "medium", "large"]:
        for variant in ["baseline", "bottleneck", "decoupled", "gqa"]:
            dir_name = f"{size}_{variant}"
            dir_path = runs_dir / dir_name
            if dir_path.exists() and (dir_path / "train.jsonl").exists():
                # Create human-readable name
                if variant == "baseline":
                    name = f"Standard ({size})"
                elif variant == "bottleneck":
                    name = f"Bottleneck ({size})"
                elif variant == "decoupled":
                    name = f"Decoupled ({size})"
                elif variant == "gqa":
                    name = f"GQA ({size})"
                experiments[name] = str(dir_path)
    
    # Also check for old-style paper_* directories
    for dir_path in runs_dir.glob("paper_*"):
        if (dir_path / "train.jsonl").exists():
            name = dir_path.name.replace("paper_", "").replace("_", " ").title()
            if name not in experiments:
                experiments[name] = str(dir_path)
    
    return experiments

PAPER_EXPERIMENTS = discover_experiments()

# Legacy mapping for backward compatibility
WIKITEXT_EXPERIMENTS = {}  # Removed - FineWeb only
FINEWEB_EXPERIMENTS = PAPER_EXPERIMENTS  # Now the main experiments

# Memory data is now read from actual measurements in train.jsonl
# No more estimates - we measure everything

OUTPUT_DIR = Path("assets")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentResult:
    name: str
    config: dict
    eval_steps: List[int]
    train_losses: List[float]
    val_losses: List[float]
    best_val: float
    total_time: float
    attn_dim: int
    
    # Measured memory (from instrumentation, not estimates)
    model_params_mb: float = 0.0
    kv_cache_train_mb: float = 0.0
    kv_cache_128k_fp16_mb: float = 0.0
    kv_cache_128k_q4_mb: float = 0.0
    compression_ratio: float = 1.0
    
    @property
    def best_ppl(self) -> float:
        import math
        return math.exp(self.best_val) if self.best_val < 20 else float('inf')

# =============================================================================
# PARSING
# =============================================================================

def parse_experiment(name: str, run_dir: str) -> Optional[ExperimentResult]:
    """Parse a JSONL log file and extract key metrics including measured memory."""
    log_path = os.path.join(run_dir, "train.jsonl")
    
    if not os.path.exists(log_path):
        print(f"  ⚠ Not found: {log_path}")
        return None
    
    config = {}
    eval_steps = []
    train_losses = []
    val_losses = []
    best_val = float('inf')
    total_time = 0.0
    
    # Memory measurements (from instrumentation, not estimates)
    model_params_mb = 0.0
    kv_cache_train_mb = 0.0
    kv_cache_128k_fp16_mb = 0.0
    kv_cache_128k_q4_mb = 0.0
    compression_ratio = 1.0
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                if data.get("type") == "run_config":
                    config = data.get("config", {})
                
                elif data.get("type") == "eval":
                    eval_steps.append(data["step"])
                    train_losses.append(data["train_loss"])
                    val_losses.append(data["val_loss"])
                
                elif data.get("type") == "best":
                    best_val = data["best_val"]
                
                elif data.get("type") == "done":
                    best_val = data.get("best_val", best_val)
                    total_time = data.get("total_seconds", 0)
                
                elif data.get("type") == "memory_measurement":
                    # Extract actual measured memory values
                    model_params = data.get("model_params", {})
                    model_params_mb = model_params.get("total_params_bytes", 0) / (1024 * 1024)
                    
                    kv_train = data.get("kv_cache_training", {})
                    kv_cache_train_mb = kv_train.get("fp16_total_mb", 0)
                    
                    kv_128k = data.get("kv_cache_128k", {})
                    kv_cache_128k_fp16_mb = kv_128k.get("fp16_total_mb", 0)
                    kv_cache_128k_q4_mb = kv_128k.get("q4_total_mb", 0)
                    compression_ratio = kv_128k.get("fp16_to_q4_ratio", 1.0)
                    
            except json.JSONDecodeError:
                continue
    
    if not eval_steps:
        print(f"  ⚠ No eval data in: {log_path}")
        return None
    
    # Determine attention dimension
    attn_dim = config.get("attn_dim", 512)
    if config.get("attn_mode") == "decoupled":
        attn_dim = config.get("sem_dim", 32) + config.get("geo_dim", 64)
    
    return ExperimentResult(
        name=name,
        config=config,
        eval_steps=eval_steps,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val=min(val_losses) if val_losses else best_val,
        total_time=total_time,
        attn_dim=attn_dim,
        model_params_mb=model_params_mb,
        kv_cache_train_mb=kv_cache_train_mb,
        kv_cache_128k_fp16_mb=kv_cache_128k_fp16_mb,
        kv_cache_128k_q4_mb=kv_cache_128k_q4_mb,
        compression_ratio=compression_ratio,
    )


def load_all_experiments() -> Tuple[Dict[str, ExperimentResult], Dict[str, ExperimentResult]]:
    """Load all experiment results."""
    print("\nLoading FineWeb-Edu experiments (paper_*)...")
    experiments = {}
    for name, path in PAPER_EXPERIMENTS.items():
        result = parse_experiment(name, path)
        if result:
            experiments[name] = result
            print(f"  ✓ {name}: best_val={result.best_val:.4f}")
    
    # Return as (wikitext, fineweb) for API compatibility, but wikitext is empty
    return {}, experiments

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_convergence_plot(
    experiments: Dict[str, ExperimentResult],
    output_path: Path,
    title: str,
    highlight_key: str = "Combined 96"
):
    """Generate a convergence plot comparing multiple experiments."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color scheme
    colors = {
        "Standard (512)": "#333333",
        "Combined 96": "#4CAF50",
        "Decoupled (32/64)": "#2196F3",
        "Bottleneck 128": "#9C27B0",
        "GQA (8Q/2KV)": "#FF9800",
        "Small (d=128)": "#F44336",
    }
    
    for name, result in experiments.items():
        color = colors.get(name, "#666666")
        linewidth = 2.5 if name == highlight_key or name == "Standard (512)" else 1.5
        linestyle = "--" if name == "Standard (512)" else "-"
        marker = "s" if name == "Standard (512)" else "o"
        
        ax.plot(
            result.eval_steps, 
            result.val_losses, 
            label=f"{name} ({result.best_val:.3f})",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            alpha=0.9 if linewidth > 2 else 0.7,
        )
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_pareto_plot(
    wikitext: Dict[str, ExperimentResult],
    fineweb: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate Pareto frontier showing memory vs quality trade-off."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # WikiText-2 points
    wt_dims = [r.attn_dim for r in wikitext.values()]
    wt_vals = [r.best_val for r in wikitext.values()]
    wt_names = list(wikitext.keys())
    
    ax.scatter(wt_dims, wt_vals, c='#2196F3', s=100, zorder=5, 
               label='WikiText-2', edgecolors='white', linewidth=1.5)
    
    for i, name in enumerate(wt_names):
        offset = (8, 5)
        if "Combined" in name:
            offset = (8, -12)
        ax.annotate(name, (wt_dims[i], wt_vals[i]), 
                   textcoords="offset points", xytext=offset,
                   fontsize=8, color='#2196F3')
    
    # FineWeb points
    if fineweb:
        fw_dims = [r.attn_dim for r in fineweb.values()]
        fw_vals = [r.best_val for r in fineweb.values()]
        fw_names = list(fineweb.keys())
        
        ax.scatter(fw_dims, fw_vals, c='#FF5722', s=100, zorder=5,
                   label='FineWeb-Edu', marker='s', edgecolors='white', linewidth=1.5)
        
        for i, name in enumerate(fw_names):
            ax.annotate(name, (fw_dims[i], fw_vals[i]),
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=8, color='#FF5722')
    
    ax.set_xlabel("Attention Dimension (d_attn)", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("Pareto Frontier: Compression vs Quality", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Memory savings annotation
    if "Standard (512)" in wikitext and "Combined 96" in wikitext:
        std = wikitext["Standard (512)"]
        comb = wikitext["Combined 96"]
        ratio = std.attn_dim / comb.attn_dim
        ax.annotate(
            f"{ratio:.1f}× memory\nreduction",
            xy=(comb.attn_dim + 20, comb.best_val),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_memory_plot(
    experiments: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate memory comparison plot using ACTUAL MEASURED data."""
    import matplotlib.pyplot as plt
    
    # Filter to experiments with memory measurements
    exps_with_memory = {k: v for k, v in experiments.items() if v.kv_cache_128k_fp16_mb > 0}
    
    if not exps_with_memory:
        print(f"  ⚠ No memory measurements found. Run experiments with instrumentation enabled.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(exps_with_memory.keys())
    fp16_vals = [exps_with_memory[n].kv_cache_128k_fp16_mb for n in names]
    q4_vals = [exps_with_memory[n].kv_cache_128k_q4_mb for n in names]
    
    # Left: FP16 KV cache at 128k
    colors = ['#4CAF50' if 'Bottleneck' in n or 'Decoupled' in n else '#2196F3' for n in names]
    bars1 = ax1.barh(names, fp16_vals, color=colors, edgecolor='white')
    ax1.set_xlabel("KV Cache Memory (MB) @ 128k tokens, FP16", fontsize=11)
    ax1.set_title("Measured KV Cache Memory (FP16)", fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    for bar, val in zip(bars1, fp16_vals):
        ax1.text(val + max(fp16_vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} MB', va='center', fontsize=9)
    
    # Right: Q4 KV cache at 128k
    bars2 = ax2.barh(names, q4_vals, color=colors, edgecolor='white')
    ax2.set_xlabel("KV Cache Memory (MB) @ 128k tokens, Q4", fontsize=11)
    ax2.set_title("Measured KV Cache Memory (Q4 Quantized)", fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    for bar, val in zip(bars2, q4_vals):
        ax2.text(val + max(q4_vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} MB', va='center', fontsize=9)
    
    # Add compression ratio annotation
    if len(names) >= 2:
        baseline_fp16 = max(fp16_vals)
        best_q4 = min(q4_vals)
        if best_q4 > 0:
            ratio = baseline_fp16 / best_q4
            ax2.annotate(f"Max compression: {ratio:.0f}×",
                        xy=(max(q4_vals) * 0.5, len(names) - 0.5), fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_comparison_bar(
    wikitext: Dict[str, ExperimentResult],
    fineweb: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate bar chart comparing final losses."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WikiText-2 (left)
    if wikitext:
        names = list(wikitext.keys())
        vals = [wikitext[n].best_val for n in names]
        colors = ['#4CAF50' if 'Combined' in n else '#2196F3' for n in names]
        
        bars = ax1.barh(names, vals, color=colors, edgecolor='white')
        ax1.set_xlabel("Best Validation Loss", fontsize=11)
        ax1.set_title("WikiText-2 Results", fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        for bar, val in zip(bars, vals):
            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
    
    # FineWeb (right)
    if fineweb:
        names = list(fineweb.keys())
        vals = [fineweb[n].best_val for n in names]
        colors = ['#FF5722' if 'Standard' in n else '#FF8A65' for n in names]
        
        bars = ax2.barh(names, vals, color=colors, edgecolor='white')
        ax2.set_xlabel("Best Validation Loss", fontsize=11)
        ax2.set_title("FineWeb-Edu Results", fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        
        for bar, val in zip(bars, vals):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        # Gap annotation
        if len(vals) >= 2:
            gap = vals[1] - vals[0]
            gap_pct = (gap / vals[0]) * 100
            ax2.annotate(f"Δ = {gap:.3f} ({gap_pct:.1f}%)",
                        xy=(max(vals) * 0.7, 0.5), fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def generate_latex_table(
    experiments: Dict[str, ExperimentResult],
    output_path: Path,
    caption: str
):
    """Generate a LaTeX table for the paper with measured memory data."""
    import math
    
    # Check if we have memory measurements
    has_memory = any(r.kv_cache_128k_fp16_mb > 0 for r in experiments.values())
    
    if has_memory:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\begin{tabular}{@{}lccccc@{}}",
            r"\toprule",
            r"Model & $d_{attn}$ & Val Loss & PPL & KV@128k (MB) & Compress. \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Model & $d_{attn}$ & Val Loss & PPL & Time (s) \\",
            r"\midrule",
        ]
    
    # Sort by best_val
    sorted_exps = sorted(experiments.items(), key=lambda x: x[1].best_val)
    best_name = sorted_exps[0][0] if sorted_exps else None
    
    for name, result in experiments.items():
        ppl = math.exp(result.best_val) if result.best_val < 20 else float('inf')
        ppl_str = f"{ppl:.1f}" if ppl < 10000 else r"$\infty$"
        
        if has_memory:
            kv_mem = f"{result.kv_cache_128k_fp16_mb:.0f}" if result.kv_cache_128k_fp16_mb > 0 else "—"
            compress = f"{result.compression_ratio:.1f}$\\times$" if result.compression_ratio > 1 else "—"
            
            if name == best_name:
                lines.append(rf"\textbf{{{name}}} & {result.attn_dim} & \textbf{{{result.best_val:.4f}}} & \textbf{{{ppl_str}}} & {kv_mem} & {compress} \\")
            else:
                lines.append(rf"{name} & {result.attn_dim} & {result.best_val:.4f} & {ppl_str} & {kv_mem} & {compress} \\")
        else:
            time_str = f"{result.total_time:.0f}" if result.total_time > 0 else "—"
            if name == best_name:
                lines.append(rf"\textbf{{{name}}} & {result.attn_dim} & \textbf{{{result.best_val:.4f}}} & \textbf{{{ppl_str}}} & {time_str} \\")
            else:
                lines.append(rf"{name} & {result.attn_dim} & {result.best_val:.4f} & {ppl_str} & {time_str} \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  → Saved: {output_path}")


def generate_summary_markdown(
    wikitext: Dict[str, ExperimentResult],
    fineweb: Dict[str, ExperimentResult],
    output_path: Path
):
    """Generate a markdown summary of all results with MEASURED memory data."""
    import math
    
    lines = [
        "# Experiment Results Summary",
        "",
        "Generated from instrumented experiment logs.",
        "**All memory values are measured, not estimated.**",
        "",
        "## Model Performance",
        "",
        "| Model | d_attn | Best Val | PPL | Params (MB) |",
        "|-------|--------|----------|-----|-------------|",
    ]
    
    for name, r in sorted(fineweb.items(), key=lambda x: x[1].best_val):
        ppl = math.exp(r.best_val)
        params = f"{r.model_params_mb:.1f}" if r.model_params_mb > 0 else "—"
        lines.append(f"| {name} | {r.attn_dim} | {r.best_val:.4f} | {ppl:.1f} | {params} |")
    
    # Memory comparison table (from actual measurements)
    has_memory = any(r.kv_cache_128k_fp16_mb > 0 for r in fineweb.values())
    if has_memory:
        lines.extend([
            "",
            "## Measured KV Cache Memory @ 128k Tokens",
            "",
            "| Model | FP16 (MB) | Q4 (MB) | Compression |",
            "|-------|-----------|---------|-------------|",
        ])
        
        for name, r in sorted(fineweb.items(), key=lambda x: x[1].kv_cache_128k_fp16_mb, reverse=True):
            if r.kv_cache_128k_fp16_mb > 0:
                fp16 = f"{r.kv_cache_128k_fp16_mb:.0f}"
                q4 = f"{r.kv_cache_128k_q4_mb:.0f}"
                comp = f"{r.compression_ratio:.1f}×"
                lines.append(f"| {name} | {fp16} | {q4} | {comp} |")
    
    # Key findings with actual numbers
    baseline_candidates = [n for n in fineweb.keys() if "Standard" in n or "baseline" in n.lower()]
    bottleneck_candidates = [n for n in fineweb.keys() if "Bottleneck" in n or "Decoupled" in n]
    
    if baseline_candidates and bottleneck_candidates:
        baseline_name = baseline_candidates[0]
        bottleneck_name = min(bottleneck_candidates, key=lambda n: fineweb[n].best_val)
        
        std = fineweb[baseline_name]
        best = fineweb[bottleneck_name]
        
        lines.extend([
            "",
            "## Key Findings",
            "",
        ])
        
        # Memory reduction (from actual measurements)
        if std.kv_cache_128k_fp16_mb > 0 and best.kv_cache_128k_fp16_mb > 0:
            mem_reduction = std.kv_cache_128k_fp16_mb / best.kv_cache_128k_fp16_mb
            lines.append(f"- **Measured Memory Reduction (FP16)**: {mem_reduction:.1f}× ({std.kv_cache_128k_fp16_mb:.0f} MB → {best.kv_cache_128k_fp16_mb:.0f} MB)")
        
        if std.kv_cache_128k_fp16_mb > 0 and best.kv_cache_128k_q4_mb > 0:
            total_reduction = std.kv_cache_128k_fp16_mb / best.kv_cache_128k_q4_mb
            lines.append(f"- **Total Memory Reduction (FP16→Q4)**: {total_reduction:.0f}× ({std.kv_cache_128k_fp16_mb:.0f} MB → {best.kv_cache_128k_q4_mb:.0f} MB)")
        
        lines.extend([
            f"- **Quality Gap**: {best.best_val - std.best_val:+.4f} loss",
            f"- **Best Bottleneck**: {bottleneck_name} (val={best.best_val:.4f})",
        ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  → Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Generating Paper Figures (FineWeb-Edu)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    _, fineweb = load_all_experiments()
    
    if not fineweb:
        print("\n⚠ No experiment data found!")
        print("Run experiments first: make paper_all")
        return
    
    print("\nGenerating figures...")
    print("-" * 40)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        # Fig 1: FineWeb Convergence
        if fineweb:
            generate_convergence_plot(
                fineweb,
                OUTPUT_DIR / "fig1_convergence.png",
                "FineWeb-Edu: Validation Loss Convergence",
                highlight_key="Bottleneck"
            )
        
        # Fig 2: Pareto
        generate_pareto_plot({}, fineweb, OUTPUT_DIR / "fig2_pareto.png")
        
        # Fig 3: Bar comparison
        generate_comparison_bar({}, fineweb, OUTPUT_DIR / "fig3_comparison_bar.png")
        
        # Fig 4: Memory comparison (from actual measurements!)
        generate_memory_plot(fineweb, OUTPUT_DIR / "fig4_memory.png")
        
    except ImportError:
        print("⚠ matplotlib not installed - skipping plots")
    
    print("\nGenerating tables...")
    print("-" * 40)
    
    # LaTeX table
    if fineweb:
        generate_latex_table(
            fineweb,
            OUTPUT_DIR / "table1_fineweb.tex", 
            "FineWeb-Edu Results (1024 context, 6000 steps)"
        )
    
    # Markdown summary
    generate_summary_markdown({}, fineweb, OUTPUT_DIR / "results_summary.md")
    
    print("\n" + "=" * 60)
    print("  Done! Check assets/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()

