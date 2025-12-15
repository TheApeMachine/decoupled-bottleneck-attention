#!/usr/bin/env python3
"""
vis_heatmap.py
Visualize attention patterns from trained models.

IMPORTANT: Before running this script, you need to make ONE small edit
to v21_transformer_decoupled_bottleneck.py (or the _gqa version):

In DecoupledBottleneckAttention.forward(), find the line:
    attn = self.drop(attn)

Add immediately after:
    self.last_attn = attn  # Save for visualization

This adds zero overhead but lets us inspect attention patterns.

Usage:
    python3 vis_heatmap.py --ckpt runs/v21_bottleneck_rope/best.pt
"""
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# We need to import from the training script
try:
    from v21_transformer_decoupled_bottleneck_gqa import GPT, ModelConfig, pick_device
except ImportError:
    try:
        from v21_transformer_decoupled_bottleneck import GPT, ModelConfig, pick_device
    except ImportError:
        print("ERROR: Could not import from v21 script. Make sure you're in the right directory.")
        sys.exit(1)


OUTPUT_DIR = Path("assets/heatmaps")


def check_attn_capture(model: GPT) -> bool:
    """Check if the model has been modified to capture attention weights."""
    # Try running a dummy forward pass and see if last_attn exists
    try:
        dummy_input = torch.zeros((1, 4), dtype=torch.long, device=next(model.parameters()).device)
        with torch.no_grad():
            model(dummy_input)
        
        # Check if any attention layer has last_attn
        for block in model.blocks:
            if hasattr(block.attn, 'last_attn'):
                return True
        return False
    except Exception:
        return False


def visualize_attention(
    model: GPT, 
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    output_path: str = None,
    run_name: str = None
) -> None:
    """
    Generate attention heatmap for a specific layer and head.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    # Get attention weights
    attn_layer = model.blocks[layer_idx].attn
    if not hasattr(attn_layer, 'last_attn'):
        print("\n" + "=" * 60)
        print("ERROR: Attention weights not captured!")
        print("=" * 60)
        print("\nYou need to modify the attention forward() method.")
        print("Find this line in DecoupledBottleneckAttention.forward():")
        print("    attn = self.drop(attn)")
        print("\nAdd after it:")
        print("    self.last_attn = attn  # Save for visualization")
        print("=" * 60)
        return
    
    # attn_map shape: (Batch, Heads, SeqLen, SeqLen)
    attn_map = attn_layer.last_attn[0, head_idx].detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Labels
    seq_len = attn_map.shape[0]
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    # Title
    mode = model.cfg.attn_mode
    name_str = run_name if run_name else mode.capitalize()
    title = f'Attention Pattern: {name_str}\n(Layer {layer_idx}, Head {head_idx})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add annotations for small sequences
    if seq_len <= 10:
        for i in range(seq_len):
            for j in range(seq_len):
                val = attn_map[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color=color, fontsize=8)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    return fig


def compare_attention_patterns(
    baseline_ckpt: str,
    bottleneck_ckpt: str,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    output_path: str = None
) -> None:
    """
    Side-by-side comparison of attention patterns from two models.
    """
    device = pick_device()
    
    # Load baseline
    print(f"Loading baseline from {baseline_ckpt}...")
    ckpt_base = torch.load(baseline_ckpt, map_location=device)
    cfg_base = ModelConfig(**ckpt_base['config'])
    model_base = GPT(cfg_base).to(device)
    model_base.load_state_dict(ckpt_base['model'])
    model_base.eval()
    
    # Load bottleneck
    print(f"Loading bottleneck from {bottleneck_ckpt}...")
    ckpt_bn = torch.load(bottleneck_ckpt, map_location=device)
    cfg_bn = ModelConfig(**ckpt_bn['config'])
    model_bn = GPT(cfg_bn).to(device)
    model_bn.load_state_dict(ckpt_bn['model'])
    model_bn.eval()
    
    input_ids = input_ids.to(device)
    
    # Forward pass
    with torch.no_grad():
        model_base(input_ids)
        model_bn(input_ids)
    
    # Check if attention is captured
    for model, name in [(model_base, "Baseline"), (model_bn, "Bottleneck")]:
        if not hasattr(model.blocks[layer_idx].attn, 'last_attn'):
            print(f"ERROR: {name} model doesn't have last_attn captured.")
            print("Add 'self.last_attn = attn' after 'attn = self.drop(attn)' in the forward method.")
            return
    
    # Get attention maps
    attn_base = model_base.blocks[layer_idx].attn.last_attn[0, head_idx].detach().cpu().numpy()
    attn_bn = model_bn.blocks[layer_idx].attn.last_attn[0, head_idx].detach().cpu().numpy()
    
    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Baseline
    im0 = axes[0].imshow(attn_base, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Baseline ({cfg_base.attn_mode})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im0, ax=axes[0])
    
    # Bottleneck
    im1 = axes[1].imshow(attn_bn, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Bottleneck ({cfg_bn.attn_mode})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = np.abs(attn_base - attn_bn)
    im2 = axes[2].imshow(diff, cmap='Reds', aspect='auto')
    axes[2].set_title('|Difference|', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    plt.colorbar(im2, ax=axes[2])
    
    # Overall title
    mae = np.mean(diff)
    fig.suptitle(f'Attention Comparison (Layer {layer_idx}, Head {head_idx}) — MAE: {mae:.4f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize attention patterns')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--head', type=int, default=0, help='Head index')
    parser.add_argument('--seq-len', type=int, default=8, help='Sequence length for visualization')
    parser.add_argument('--compare', type=str, default=None, 
                        help='Path to second checkpoint for comparison')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for output file (defaults to run directory name)')
    args = parser.parse_args()
    
    device = pick_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt['config'])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"Model config: attn_mode={cfg.attn_mode}, d_model={cfg.d_model}")
    if cfg.attn_mode == 'decoupled':
        print(f"  sem_dim={cfg.sem_dim}, geo_dim={cfg.geo_dim}")
    elif cfg.attn_mode in ('bottleneck', 'gqa'):
        print(f"  attn_dim={cfg.attn_dim}")
    
    # Generate dummy input (sequential IDs for clarity)
    input_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    
    # Determine output name: use --name, or derive from checkpoint path
    if args.name:
        run_name = args.name
    else:
        # Extract run directory name from checkpoint path (e.g., "runs/v21_baseline/best.pt" -> "v21_baseline")
        ckpt_path = Path(args.ckpt)
        run_name = ckpt_path.parent.name
    
    if args.compare:
        # Comparison mode
        output_path = OUTPUT_DIR / f"{run_name}_comparison_L{args.layer}_H{args.head}.png"
        compare_attention_patterns(
            args.ckpt, args.compare, input_ids,
            layer_idx=args.layer, head_idx=args.head,
            output_path=str(output_path)
        )
    else:
        # Single model visualization
        output_path = OUTPUT_DIR / f"{run_name}_L{args.layer}_H{args.head}.png"
        visualize_attention(
            model, input_ids,
            layer_idx=args.layer, head_idx=args.head,
            output_path=str(output_path),
            run_name=run_name
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

