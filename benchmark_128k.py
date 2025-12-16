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
    
    # Override block size to allow extrapolation
    if model.cfg.block_size < context_length:
        print(f"  [Extending block_size: {model.cfg.block_size} -> {context_length}]")
        model.cfg.block_size = context_length
        # Note: We do NOT resize self.causal_mask because creating a 128k x 128k bool tensor
        # exceeds INT_MAX elements (2^31) which crashes MPS/CUDA on some setups.
        # Since we use chunked inference (chunk_size=1024), we never need a mask larger
        # than the chunk size, so the original mask (e.g. 1024x1024) is sufficient
        # as long as we don't run a full dense forward pass.

    input_ids = tokens[:context_length].unsqueeze(0).to(device)
    target_ids = tokens[1:context_length + 1].unsqueeze(0).to(device)
    
    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...", end=" ", flush=True)
    # We skip full-context warmup to avoid OOM. 
    # Instead we just run a small dummy pass to wake up the GPU.
    dummy_input = input_ids[:, :128]
    for _ in range(warmup_runs):
        with torch.no_grad():
             _ = model(dummy_input)
    
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
    
    # We must use chunked inference to avoid OOM on the full 128k matrix
    # processing the sequence in blocks of 'chunk_size' using the KV cache.
    chunk_size = 1024 
    
    for _ in range(bench_runs):
        # Measure memory before
        mem_before = get_memory_usage_gb()
        
        # Reset model cache
        caches = []
        # Initialize caches (we rely on model.generate structure or manually build them)
        # Actually, let's just use the model's forward pass with cache in a loop.
        
        total_loss = 0.0
        total_time = 0.0
        
        # Initial chunk
        curr_pos = 0
        caches = None # Start with no cache
        
        # We process the prompt in chunks. 
        # For the first chunk, we compute standard attention.
        # For subsequent chunks, we use the cache.
        
        # NOTE: This effectively benchmarks "Prefill" speed if we pass large chunks,
        # or "Decode" speed if we pass 1 token at a time.
        # To measure 128k context support, we need to process the whole prompt
        # and see if it fits.
        
        # If we just want to test "can it handle 128k context inference",
        # we should process (Context-1) tokens to fill cache, then measure generation of last token.
        
        try:
            # 1. Fill Cache (Prefill) in chunks to avoid O(N^2) OOM
            # We don't time the prefill of the whole 128k for the "inference latency" metric,
            # we want to measure the time to generate *at* 128k.
            
            with torch.no_grad():
                # We need to build up the cache up to context_length - 1
                # We can do this in chunks of 4096 to be safe.
                prefill_chunk = 4096
                caches = None
                
                # We only need to process up to the last token to measure the *next* token prediction
                # But to measure perplexity, we need loss on all tokens.
                # Let's separate the two:
                # 1. Perplexity: Sum of losses across chunks.
                # 2. Latency: Time to forward the last chunk.
                
                # Simplification: We will run the *last chunk* of size 128 to measure 
                # performance at deep context.
                
                # A. Fast-forward cache (simulated) or just run forward on full sequence?
                # We can't run full sequence. We must chunk.
                
                start_prefill = time.perf_counter()
                
                # Iterate through chunks
                for i in range(0, context_length, chunk_size):
                    # End index for this chunk
                    end = min(i + chunk_size, context_length)
                    
                    chunk_input = input_ids[:, i:end]
                    
                    # Forward pass with cache
                    # Note: v21 model forward signature: forward(idx, caches=None, pos_offset=0)
                    chunk_logits, caches = model(chunk_input, caches=caches, pos_offset=i)
                    
                    # Compute loss for this chunk
                    # Targets match input shifted by 1. 
                    # target_ids is tokens[1:context+1]
                    # chunk_target corresponds to target_ids[:, i:end]
                    chunk_target = target_ids[:, i:end]
                    
                    loss = torch.nn.functional.cross_entropy(
                        chunk_logits.view(-1, chunk_logits.size(-1)),
                        chunk_target.view(-1),
                        reduction='sum' # Sum so we can average correctly later
                    )
                    total_loss += loss.item()
                    
                    # Delete logits to save memory
                    del chunk_logits
                    
                end_prefill = time.perf_counter()
                
                # Measure single-token generation time at full context
                # This is the "Inference Time" at 128k.
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                # Generate 1 token
                # Feed the last token again? No, feed a dummy token or next token
                dummy_next = target_ids[:, -1:] 
                _, _ = model(dummy_next, caches=caches, pos_offset=context_length)
                
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                times.append(t1 - t0)
                losses.append(total_loss / context_length)

        except RuntimeError as e:
            if "out of memory" in str(e) or "buffer size" in str(e):
                print(f"OOM at context {context_length}")
                return None
            raise e
            
    print("done")
    
    # Compute metrics
    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    # tokens_per_sec is now 1 / avg_time (since we timed 1 token generation)
    tokens_per_sec = 1.0 / avg_time 
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
    cache_elements = 2 * cfg.n_layer * context_length * kv_dim
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
            print(f"Trained context: {cfg.block_size}")
            
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
        print(f"Trained context: {cfg.block_size}")
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
            train_ctx = cfg.block_size
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
            ax.axvline(x=cfg.block_size, color='green', linestyle='--', 
                       label=f'Train context ({cfg.block_size})')
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

