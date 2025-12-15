#!/usr/bin/env python3
"""
benchmark_128k.py
Direct 128k context inference benchmark.

This script measures:
1. Actual memory usage at 128k context
2. Inference throughput (tokens/second)
3. Perplexity at 128k context
4. Comparison between Decoupled and Baseline architectures

Requires: MacBook Pro M4 Max with 128GB unified memory

Usage:
    python3 benchmark_128k.py --ckpt runs/context_1024/best.pt --context 131072
    python3 benchmark_128k.py --compare runs/context_1024/best.pt runs/baseline_context_1024/best.pt
"""
import argparse
import torch
import time
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Import from training script
try:
    from v21_transformer_decoupled_bottleneck_gqa import GPT, ModelConfig, pick_device
except ImportError:
    print("ERROR: Could not import from v21 script.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB (MPS/CUDA)."""
    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, estimate from system
        import subprocess
        result = subprocess.run(
            ['vm_stat'], capture_output=True, text=True
        )
        # Parse vm_stat output (rough approximation)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Pages active' in line:
                pages = int(line.split(':')[1].strip().rstrip('.'))
                return (pages * 16384) / (1024**3)  # 16KB pages to GB
        return 0.0
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def load_tokens(path: str, max_tokens: int = 200_000) -> torch.Tensor:
    """Load token file."""
    with open(path, 'r') as f:
        content = f.read()
    
    try:
        tokens = [int(t) for t in content.split()[:max_tokens]]
        return torch.tensor(tokens, dtype=torch.long)
    except ValueError:
        print(f"Could not parse {path} as token file")
        sys.exit(1)


@dataclass
class BenchmarkResult:
    context_length: int
    model_name: str
    attn_mode: str
    memory_gb: float
    inference_time_ms: float
    tokens_per_sec: float
    loss: float
    perplexity: float


@torch.no_grad()
def benchmark_inference(model: GPT, tokens: torch.Tensor,
                        context_length: int, device: str,
                        warmup_runs: int = 2,
                        bench_runs: int = 5) -> BenchmarkResult:
    """
    Benchmark inference at a specific context length.
    """
    model.eval()
    
    # Prepare input
    if len(tokens) < context_length + 1:
        print(f"Warning: Not enough tokens ({len(tokens)}) for context {context_length}")
        return None
    
    input_ids = tokens[:context_length].unsqueeze(0).to(device)
    target_ids = tokens[1:context_length + 1].unsqueeze(0).to(device)
    
    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
    for _ in range(warmup_runs):
        _ = model(input_ids)
    
    # Synchronize
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("done")
    
    # Benchmark
    print(f"  Benchmarking ({bench_runs} runs)...", end=" ", flush=True)
    times = []
    losses = []
    
    for _ in range(bench_runs):
        # Measure memory before
        mem_before = get_memory_usage_gb()
        
        # Time the forward pass
        start = time.perf_counter()
        logits, _ = model(input_ids)
        
        # Synchronize for accurate timing
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        losses.append(loss.item())
    
    print("done")
    
    # Compute metrics
    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    tokens_per_sec = context_length / avg_time
    memory_gb = get_memory_usage_gb()
    perplexity = math.exp(avg_loss)
    
    return BenchmarkResult(
        context_length=context_length,
        model_name=model.cfg.attn_mode,
        attn_mode=model.cfg.attn_mode,
        memory_gb=memory_gb,
        inference_time_ms=avg_time * 1000,
        tokens_per_sec=tokens_per_sec,
        loss=avg_loss,
        perplexity=perplexity
    )


def theoretical_kv_cache_size(cfg: ModelConfig, context_length: int, 
                               quantization_bits: int = 16) -> float:
    """Calculate theoretical KV cache size in GB."""
    if cfg.attn_mode == 'decoupled':
        kv_dim = cfg.sem_dim + cfg.geo_dim
    elif cfg.attn_mode == 'bottleneck':
        kv_dim = cfg.attn_dim
    else:  # standard
        kv_dim = cfg.d_model
    
    # K + V, for all layers
    cache_elements = 2 * cfg.layers * context_length * kv_dim
    cache_bits = cache_elements * quantization_bits
    cache_gb = cache_bits / 8 / (1024**3)
    
    return cache_gb


def main():
    parser = argparse.ArgumentParser(description='128k Context Benchmark')
    parser.add_argument('--ckpt', type=str, help='Model checkpoint')
    parser.add_argument('--compare', type=str, nargs=2, 
                        help='Compare two checkpoints')
    parser.add_argument('--data', type=str, default='fineweb_100m.tokens',
                        help='Token file')
    parser.add_argument('--contexts', type=int, nargs='+',
                        default=[1024, 4096, 16384, 65536, 131072],
                        help='Context lengths to benchmark')
    parser.add_argument('--output', type=str, default='assets/benchmark_128k.png',
                        help='Output plot path')
    args = parser.parse_args()
    
    device = pick_device()
    print(f"Device: {device}")
    print(f"Contexts to benchmark: {args.contexts}")
    
    # Load tokens
    print(f"\nLoading tokens from: {args.data}")
    max_tokens = max(args.contexts) + 1000
    tokens = load_tokens(args.data, max_tokens)
    print(f"Loaded {len(tokens):,} tokens")
    
    if args.compare:
        # Comparison mode
        ckpts = args.compare
        all_results = {}
        
        for ckpt_path in ckpts:
            print(f"\n{'='*60}")
            print(f"Loading: {ckpt_path}")
            print('='*60)
            
            ckpt = torch.load(ckpt_path, map_location=device)
            cfg = ModelConfig(**ckpt['config'])
            model = GPT(cfg).to(device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            
            model_name = f"{cfg.attn_mode}"
            if cfg.attn_mode == 'decoupled':
                model_name += f" ({cfg.sem_dim}/{cfg.geo_dim})"
            elif cfg.attn_mode == 'bottleneck':
                model_name += f" ({cfg.attn_dim})"
            else:
                model_name += f" ({cfg.d_model})"
            
            print(f"Model: {model_name}")
            print(f"Trained context: {cfg.block}")
            
            results = []
            for ctx in args.contexts:
                if ctx > len(tokens) - 1:
                    print(f"Skipping {ctx} (not enough tokens)")
                    continue
                
                print(f"\nContext {ctx:,}:")
                try:
                    result = benchmark_inference(model, tokens, ctx, device)
                    if result:
                        results.append(result)
                        kv_cache = theoretical_kv_cache_size(cfg, ctx)
                        print(f"  → Time: {result.inference_time_ms:.1f}ms")
                        print(f"  → Throughput: {result.tokens_per_sec:,.0f} tok/s")
                        print(f"  → Loss: {result.loss:.4f} (PPL: {result.perplexity:.1f})")
                        print(f"  → Theoretical KV cache: {kv_cache:.3f} GB")
                except Exception as e:
                    print(f"  → FAILED: {e}")
            
            all_results[model_name] = results
            
            # Clear model from memory
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print comparison summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        for ctx in args.contexts:
            print(f"\nContext: {ctx:,}")
            print("-" * 60)
            for model_name, results in all_results.items():
                result = next((r for r in results if r.context_length == ctx), None)
                if result:
                    print(f"  {model_name:<30} Loss: {result.loss:.4f}  "
                          f"Time: {result.inference_time_ms:.1f}ms  "
                          f"Tok/s: {result.tokens_per_sec:,.0f}")
        
    else:
        # Single model benchmark
        if not args.ckpt:
            print("Error: Provide --ckpt or --compare")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Loading: {args.ckpt}")
        print('='*60)
        
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ModelConfig(**ckpt['config'])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        print(f"Attention mode: {cfg.attn_mode}")
        print(f"Trained context: {cfg.block}")
        if cfg.attn_mode == 'decoupled':
            print(f"Dimensions: sem={cfg.sem_dim}, geo={cfg.geo_dim}")
        
        results = []
        for ctx in args.contexts:
            if ctx > len(tokens) - 1:
                print(f"Skipping {ctx} (not enough tokens)")
                continue
            
            print(f"\nContext {ctx:,}:")
            try:
                result = benchmark_inference(model, tokens, ctx, device)
                if result:
                    results.append(result)
                    kv_cache = theoretical_kv_cache_size(cfg, ctx)
                    kv_cache_q4 = theoretical_kv_cache_size(cfg, ctx, 4)
                    
                    print(f"  → Inference time: {result.inference_time_ms:.1f}ms")
                    print(f"  → Throughput: {result.tokens_per_sec:,.0f} tok/s")
                    print(f"  → Loss: {result.loss:.4f}")
                    print(f"  → Perplexity: {result.perplexity:.1f}")
                    print(f"  → KV cache (FP16): {kv_cache:.3f} GB")
                    print(f"  → KV cache (Q4): {kv_cache_q4:.3f} GB")
            except Exception as e:
                print(f"  → FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary table
        if results:
            print("\n" + "=" * 80)
            print("BENCHMARK SUMMARY")
            print("=" * 80)
            print(f"{'Context':<12} {'Time (ms)':<12} {'Tok/s':<12} {'Loss':<10} {'PPL':<10}")
            print("-" * 80)
            
            for r in results:
                print(f"{r.context_length:<12,} {r.inference_time_ms:<12.1f} "
                      f"{r.tokens_per_sec:<12,.0f} {r.loss:<10.4f} {r.perplexity:<10.1f}")
            
            # Extrapolation quality
            train_ctx = cfg.block
            train_result = next((r for r in results if r.context_length == train_ctx), None)
            max_result = results[-1]
            
            if train_result and max_result.context_length > train_ctx:
                ppl_degradation = ((max_result.perplexity - train_result.perplexity) 
                                   / train_result.perplexity) * 100
                extrap_ratio = max_result.context_length / train_ctx
                
                print("\n" + "-" * 80)
                print(f"Extrapolation: {train_ctx} → {max_result.context_length} ({extrap_ratio:.0f}x)")
                print(f"PPL degradation: {ppl_degradation:+.1f}%")
                
                if ppl_degradation < 50:
                    print("✓ EXCELLENT: RoPE extrapolation successful!")
                elif ppl_degradation < 100:
                    print("⚠ MODERATE: Some degradation at long context")
                else:
                    print("✗ POOR: Significant degradation, may need position interpolation")
        
        # Generate plot
        if HAS_MATPLOTLIB and results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            contexts = [r.context_length for r in results]
            
            # Time vs Context
            ax = axes[0, 0]
            ax.plot(contexts, [r.inference_time_ms for r in results], 'bo-', linewidth=2)
            ax.set_xlabel('Context Length')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Inference Time vs Context')
            ax.set_xscale('log', base=2)
            ax.grid(True, alpha=0.3)
            
            # Throughput vs Context
            ax = axes[0, 1]
            ax.plot(contexts, [r.tokens_per_sec for r in results], 'go-', linewidth=2)
            ax.set_xlabel('Context Length')
            ax.set_ylabel('Tokens/Second')
            ax.set_title('Throughput vs Context')
            ax.set_xscale('log', base=2)
            ax.grid(True, alpha=0.3)
            
            # PPL vs Context
            ax = axes[1, 0]
            ax.plot(contexts, [r.perplexity for r in results], 'ro-', linewidth=2)
            ax.axvline(x=cfg.block, color='green', linestyle='--', 
                       label=f'Train context ({cfg.block})')
            ax.set_xlabel('Context Length')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity vs Context (RoPE Extrapolation)')
            ax.set_xscale('log', base=2)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # KV Cache Size
            ax = axes[1, 1]
            kv_fp16 = [theoretical_kv_cache_size(cfg, c, 16) for c in contexts]
            kv_q4 = [theoretical_kv_cache_size(cfg, c, 4) for c in contexts]
            ax.plot(contexts, kv_fp16, 'b-', linewidth=2, label='FP16')
            ax.plot(contexts, kv_q4, 'g-', linewidth=2, label='Q4')
            ax.axhline(y=128, color='red', linestyle='--', label='128GB RAM')
            ax.set_xlabel('Context Length')
            ax.set_ylabel('KV Cache Size (GB)')
            ax.set_title('KV Cache Memory Requirements')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'128k Context Benchmark: {cfg.attn_mode}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved: {args.output}")
            plt.close()


if __name__ == "__main__":
    main()

