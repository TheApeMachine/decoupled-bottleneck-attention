#!/usr/bin/env python3
"""
test_needle_haystack.py
Needle-in-a-Haystack retrieval test for long-context evaluation.

This test:
1. Inserts a unique "needle" (key fact) at various depths in a long context
2. Asks the model to retrieve/use that information
3. Measures retrieval accuracy across context lengths and depths

This is the standard benchmark for evaluating long-context capabilities.

Usage:
    python3 test_needle_haystack.py \
        --ckpt runs/context_1024/best.pt \
        --depths 0.1 0.25 0.5 0.75 0.9 \
        --context-lengths 512 1024 2048 4096
"""
import argparse
import unittest
try:
    import torch
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for this module but is not available: {e}")
import random
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
    raise unittest.SkipTest("v21_transformer_decoupled_bottleneck_gqa import failed; skipping long-context script module")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Simple vocabulary for needle-haystack test
# We'll use token IDs directly since we don't have the original tokenizer
NEEDLE_PATTERNS = [
    # (needle_tokens, query_tokens, answer_token)
    # Pattern: "The secret number is X" ... "What is the secret number?"
    # We'll encode this as specific token sequences
    ([100, 200, 300], [100, 200], 300),  # Simplified: pattern A B C, query A B, answer C
    ([400, 500, 600], [400, 500], 600),
    ([700, 800, 900], [700, 800], 900),
]


def create_haystack(length: int, needle_tokens: list, depth: float, 
                    vocab_size: int = 50257) -> tuple[torch.Tensor, int]:
    """
    Create a context with a needle inserted at the specified depth.
    
    Args:
        length: Total context length
        needle_tokens: The needle to insert
        depth: Where to insert (0.0 = beginning, 1.0 = end)
        vocab_size: Vocabulary size for random filler
    
    Returns:
        (context_tensor, needle_position)
    """
    # Create random filler tokens (avoiding special tokens)
    filler = torch.randint(1000, vocab_size - 1000, (length,))
    
    # Calculate insertion position
    needle_len = len(needle_tokens)
    max_pos = length - needle_len - 10  # Leave room at end
    insert_pos = int(max_pos * depth)
    insert_pos = max(10, insert_pos)  # Don't insert at very start
    
    # Insert needle
    needle_tensor = torch.tensor(needle_tokens, dtype=torch.long)
    context = torch.cat([
        filler[:insert_pos],
        needle_tensor,
        filler[insert_pos + needle_len:]
    ])
    
    return context[:length], insert_pos


@torch.no_grad()
def test_retrieval(model: GPT, context: torch.Tensor, 
                   query_tokens: list, expected_answer: int,
                   device: str) -> tuple[bool, float, int]:
    """
    Test if the model can retrieve information from context.
    
    Returns:
        (success, confidence, predicted_token)
    """
    model.eval()
    
    # Append query to context
    query_tensor = torch.tensor(query_tokens, dtype=torch.long)
    full_input = torch.cat([context, query_tensor]).unsqueeze(0).to(device)
    
    # Get model prediction
    logits, _ = model(full_input)
    
    # Get the logits for the last position (predicting answer)
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    # Get top prediction
    top_prob, top_idx = probs.max(dim=-1)
    predicted = top_idx.item()
    
    # Check if correct
    success = (predicted == expected_answer)
    confidence = probs[expected_answer].item()
    
    return success, confidence, predicted


def run_needle_haystack(model: GPT, device: str,
                        context_lengths: list, depths: list,
                        num_trials: int = 10) -> dict:
    """
    Run full needle-in-haystack evaluation.
    
    Returns:
        Dictionary with results matrix
    """
    results = {}
    
    for ctx_len in context_lengths:
        results[ctx_len] = {}
        
        for depth in depths:
            successes = 0
            total_confidence = 0.0
            
            for trial in range(num_trials):
                # Pick a random needle pattern
                needle_tokens, query_tokens, answer = random.choice(NEEDLE_PATTERNS)
                
                # Create context with needle
                context, pos = create_haystack(ctx_len, needle_tokens, depth)
                
                # Test retrieval
                try:
                    success, confidence, _ = test_retrieval(
                        model, context, query_tokens, answer, device
                    )
                    if success:
                        successes += 1
                    total_confidence += confidence
                except Exception as e:
                    print(f"  Error at ctx={ctx_len}, depth={depth}: {e}")
            
            accuracy = successes / num_trials
            avg_confidence = total_confidence / num_trials
            
            results[ctx_len][depth] = {
                'accuracy': accuracy,
                'confidence': avg_confidence,
                'trials': num_trials
            }
            
            print(f"  ctx={ctx_len}, depth={depth:.1%}: "
                  f"acc={accuracy:.0%}, conf={avg_confidence:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Needle-in-a-Haystack test')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--depths', type=float, nargs='+', 
                        default=[0.1, 0.25, 0.5, 0.75, 0.9],
                        help='Depths to test (0=start, 1=end)')
    parser.add_argument('--context-lengths', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096],
                        help='Context lengths to test')
    parser.add_argument('--trials', type=int, default=20,
                        help='Trials per (context, depth) pair')
    parser.add_argument('--output', type=str, default='assets/needle_haystack.png',
                        help='Output plot path')
    args = parser.parse_args()
    
    device = pick_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt['config'])
    
    print(f"Model trained at context: {cfg.block_size}")
    print(f"Attention mode: {cfg.attn_mode}")
    
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Extend block_size to handle the largest context we'll test
    max_ctx = max(args.context_lengths) + 10  # +10 for query tokens
    if cfg.block_size < max_ctx:
        print(f"Extending block_size: {cfg.block_size} -> {max_ctx}")
        model.cfg.block_size = max_ctx
        # Update causal mask (only if it's not too large for INT_MAX)
        if max_ctx <= 16384:  # Safe limit for dense mask
            model.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_ctx, max_ctx, dtype=torch.bool, device=device)).view(1, 1, max_ctx, max_ctx),
                persistent=False,
            )
    
    # Run needle-haystack test
    print("\n" + "=" * 60)
    print("Needle-in-a-Haystack Test")
    print("=" * 60)
    print(f"Context lengths: {args.context_lengths}")
    print(f"Depths: {[f'{d:.0%}' for d in args.depths]}")
    print(f"Trials per cell: {args.trials}")
    print()
    
    results = run_needle_haystack(
        model, device,
        args.context_lengths, args.depths,
        num_trials=args.trials
    )
    
    # Print summary table
    print("\n" + "=" * 60)
    print("Results Summary (Accuracy %)")
    print("=" * 60)
    
    # Header
    header = f"{'Context':<10}" + "".join(f"{d:.0%}".center(10) for d in args.depths)
    print(header)
    print("-" * len(header))
    
    for ctx_len in args.context_lengths:
        row = f"{ctx_len:<10}"
        for depth in args.depths:
            acc = results[ctx_len][depth]['accuracy'] * 100
            row += f"{acc:>6.0f}%   "
        print(row)
    
    print("-" * 60)
    
    # Analyze
    train_context = cfg.block_size
    print(f"\nTrained context: {train_context}")
    
    within_train = [ctx for ctx in args.context_lengths if ctx <= train_context]
    beyond_train = [ctx for ctx in args.context_lengths if ctx > train_context]
    
    if within_train:
        avg_acc = sum(
            results[ctx][d]['accuracy'] 
            for ctx in within_train for d in args.depths
        ) / (len(within_train) * len(args.depths))
        print(f"Avg accuracy (within train): {avg_acc:.0%}")
    
    if beyond_train:
        avg_acc = sum(
            results[ctx][d]['accuracy'] 
            for ctx in beyond_train for d in args.depths
        ) / (len(beyond_train) * len(args.depths))
        print(f"Avg accuracy (extrapolation): {avg_acc:.0%}")
    
    # Generate heatmap
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create matrix
        matrix = []
        for ctx_len in args.context_lengths:
            row = [results[ctx_len][d]['accuracy'] * 100 for d in args.depths]
            matrix.append(row)
        matrix = np.array(matrix)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Labels
        ax.set_xticks(range(len(args.depths)))
        ax.set_xticklabels([f'{d:.0%}' for d in args.depths])
        ax.set_yticks(range(len(args.context_lengths)))
        ax.set_yticklabels(args.context_lengths)
        
        ax.set_xlabel('Needle Depth (0%=start, 100%=end)', fontsize=12)
        ax.set_ylabel('Context Length', fontsize=12)
        ax.set_title('Needle-in-a-Haystack: Retrieval Accuracy (%)', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', fontsize=10)
        
        # Add text annotations
        for i in range(len(args.context_lengths)):
            for j in range(len(args.depths)):
                val = matrix[i, j]
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        color=color, fontsize=10)
        
        # Mark extrapolation region
        if train_context in args.context_lengths:
            train_idx = args.context_lengths.index(train_context)
            ax.axhline(y=train_idx + 0.5, color='blue', linestyle='--', 
                       linewidth=2, label=f'Train context ({train_context})')
            ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot: {args.output}")
        plt.close()
    
    # Save CSV
    csv_path = Path(args.output).with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("context_length," + ",".join(f"depth_{d:.2f}" for d in args.depths) + "\n")
        for ctx_len in args.context_lengths:
            row = [str(ctx_len)]
            for depth in args.depths:
                row.append(f"{results[ctx_len][depth]['accuracy']:.2f}")
            f.write(",".join(row) + "\n")
    print(f"✓ Saved data: {csv_path}")


if __name__ == "__main__":
    main()

