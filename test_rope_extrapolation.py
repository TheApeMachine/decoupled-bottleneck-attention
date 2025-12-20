#!/usr/bin/env python3
"""
test_rope_extrapolation.py
Test RoPE position encoding extrapolation to unseen context lengths.

This script:
1. Loads a model trained at context length N
2. Evaluates perplexity at context lengths N, 2N, 4N, 8N
3. Measures how well RoPE generalizes to longer contexts

If perplexity degrades gracefully (not exponentially), RoPE extrapolation works.

Usage:
    python3 test_rope_extrapolation.py \
        --ckpt runs/context_1024/best.pt \
        --data fineweb_100m.tokens \
        --contexts 1024 2048 4096 8192
"""
import argparse
import unittest
try:
    import torch
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for this module but is not available: {e}")
import math
from pathlib import Path
import sys

# Import from training script
try:
    from v21_transformer_decoupled_bottleneck_gqa import GPT, ModelConfig, pick_device
except ImportError:
    if __name__ == "__main__":  # pragma: no cover
        print("ERROR: Could not import from v21 script.")
        sys.exit(1)
    raise unittest.SkipTest("v21_transformer_decoupled_bottleneck_gqa import failed; skipping RoPE extrapolation script module")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_tokens(path: str, max_tokens: int = 10_000_000) -> torch.Tensor:
    """Load token file."""
    with open(path, 'r') as f:
        content = f.read()
    
    # Try parsing as space-separated integers
    try:
        tokens = [int(t) for t in content.split()[:max_tokens]]
        return torch.tensor(tokens, dtype=torch.long)
    except ValueError:
        print(f"Could not parse {path} as token file")
        sys.exit(1)


@torch.no_grad()
def evaluate_perplexity(model: GPT, tokens: torch.Tensor, 
                        context_length: int, device: str,
                        num_batches: int = 50) -> tuple[float, float]:
    """
    Evaluate perplexity at a specific context length.
    
    Returns:
        (loss, perplexity)
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    # Create batches
    n_tokens = len(tokens)
    stride = context_length
    
    for i in range(num_batches):
        start = (i * stride) % (n_tokens - context_length - 1)
        end = start + context_length + 1
        
        if end > n_tokens:
            break
        
        batch = tokens[start:end].unsqueeze(0).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Test RoPE extrapolation')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Token file')
    parser.add_argument('--contexts', type=int, nargs='+', default=[1024, 2048, 4096, 8192],
                        help='Context lengths to test')
    parser.add_argument('--output', type=str, default='assets/rope_extrapolation.png',
                        help='Output plot path')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='Number of batches per context length')
    args = parser.parse_args()
    
    device = pick_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt['config'])
    
    print(f"Model trained at context: {cfg.block_size}")
    print(f"Attention mode: {cfg.attn_mode}")
    print(f"attn_dim: {cfg.attn_dim}")
    if cfg.attn_mode == 'decoupled':
        print(f"  d_sem={cfg.sem_dim}, d_geo={cfg.geo_dim}")
    
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Load tokens
    print(f"\nLoading tokens from: {args.data}")
    tokens = load_tokens(args.data)
    print(f"Loaded {len(tokens):,} tokens")
    
    # Test each context length
    print("\n" + "=" * 60)
    print("RoPE Extrapolation Test")
    print("=" * 60)
    
    results = []
    train_context = cfg.block_size
    
    for ctx in args.contexts:
        if ctx > len(tokens) - 1:
            print(f"Skipping {ctx} (not enough tokens)")
            continue
        
        # Temporarily modify model's block size for inference
        # We need to update both the config and the causal mask
        if ctx > model.cfg.block_size:
            model.cfg.block_size = ctx
            # Re-generate causal mask for larger context
            model.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)).view(1, 1, ctx, ctx).to(device),
                persistent=False,
            )
        
        print(f"\nEvaluating at context={ctx}...", end=" ", flush=True)
        
        try:
            loss, ppl = evaluate_perplexity(model, tokens, ctx, device, args.num_batches)
            
            ratio = ctx / train_context
            extrapolation = "✓ within train" if ratio <= 1 else f"↗ {ratio:.1f}x extrapolation"
            
            print(f"loss={loss:.4f}, ppl={ppl:.2f} ({extrapolation})")
            results.append({
                'context': ctx,
                'loss': loss,
                'ppl': ppl,
                'ratio': ratio
            })
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Summary
    print("\n" + "-" * 60)
    print(f"{'Context':<12} {'Loss':<12} {'PPL':<12} {'Extrap.':<15}")
    print("-" * 60)
    
    baseline_ppl = results[0]['ppl'] if results else 0
    for r in results:
        extrap = f"{r['ratio']:.1f}x" if r['ratio'] > 1 else "baseline"
        degradation = ((r['ppl'] - baseline_ppl) / baseline_ppl) * 100 if baseline_ppl else 0
        deg_str = f"+{degradation:.1f}%" if degradation > 0 else "—"
        print(f"{r['context']:<12} {r['loss']:<12.4f} {r['ppl']:<12.2f} {extrap:<10} {deg_str}")
    
    print("-" * 60)
    
    # Analyze extrapolation quality
    if len(results) >= 2:
        train_result = next((r for r in results if r['ratio'] <= 1), results[0])
        extrap_results = [r for r in results if r['ratio'] > 1]
        
        if extrap_results:
            max_extrap = max(extrap_results, key=lambda x: x['ratio'])
            degradation = ((max_extrap['ppl'] - train_result['ppl']) / train_result['ppl']) * 100
            
            print(f"\nExtrapolation Quality:")
            print(f"  Train context: {train_context}")
            print(f"  Max tested: {max_extrap['context']} ({max_extrap['ratio']:.1f}x)")
            print(f"  PPL degradation: {degradation:.1f}%")
            
            if degradation < 20:
                print("  → EXCELLENT: RoPE extrapolates well (<20% degradation)")
            elif degradation < 50:
                print("  → GOOD: Moderate degradation (20-50%)")
            else:
                print("  → POOR: Significant degradation (>50%)")
    
    # Generate plot
    if HAS_MATPLOTLIB and results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        contexts = [r['context'] for r in results]
        losses = [r['loss'] for r in results]
        ppls = [r['ppl'] for r in results]
        
        # Loss plot
        ax1.plot(contexts, losses, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=train_context, color='green', linestyle='--', 
                    label=f'Train context ({train_context})')
        ax1.set_xlabel('Context Length', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Loss vs Context Length', fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PPL plot (log scale)
        ax2.plot(contexts, ppls, 'ro-', linewidth=2, markersize=8)
        ax2.axvline(x=train_context, color='green', linestyle='--',
                    label=f'Train context ({train_context})')
        ax2.set_xlabel('Context Length', fontsize=12)
        ax2.set_ylabel('Perplexity (log scale)', fontsize=12)
        ax2.set_title('Perplexity vs Context Length', fontsize=14, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add extrapolation region shading
        if train_context < max(contexts):
            ax1.axvspan(train_context, max(contexts), alpha=0.1, color='red',
                        label='Extrapolation region')
            ax2.axvspan(train_context, max(contexts), alpha=0.1, color='red')
        
        plt.suptitle(f'RoPE Extrapolation: Trained at {train_context}, tested up to {max(contexts)}',
                     fontsize=14)
        plt.tight_layout()
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {args.output}")
        plt.close()
    
    # Save CSV
    csv_path = Path(args.output).with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("context,loss,ppl,ratio\n")
        for r in results:
            f.write(f"{r['context']},{r['loss']:.4f},{r['ppl']:.2f},{r['ratio']:.2f}\n")
    print(f"✓ Saved data: {csv_path}")


if __name__ == "__main__":
    main()

