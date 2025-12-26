# caramba

*A substrate for architecture research*

---

Architectures are graphs. Graphs are manifests. Running experiments should require nothing more than a YAML file.

caramba delivers a frictionless research environment without compromise—explicit building blocks, strict validation, optimized execution. You don't need to care about the low-level details. Unless you want to.

## The Pipeline

```
manifest → parse → lower → validate → build → run → verify → benchmark → artifacts
```

Every experiment flows through this chain. No magic, no hidden state, no surprises.

## Module Map

| Layer | Purpose |
|-------|---------|
| `config/` | Typed config models, preset manifests |
| `topology/` | Graph nodes—stacked, residual, nested |
| `layer/` | Thin torch modules, one concept per file |
| `cache/` | KV-cache with quantization support |
| `infer/` | Generation loop with cache management |
| `loader/` | Checkpoint readers, Llama upcycle logic |
| `trainer/` | Orchestration—upcycle, blockwise distillation |
| `benchmark/` | Perplexity, latency, memory measurement |
| `experiment/` | Unified pipeline orchestration |
| `compiler/` | Manifest → executable plan |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run full experiment (upcycle + benchmarks + artifacts)
python3 -m caramba run caramba/config/presets/llama32_1b_dba.yml --group paper

# Quick validation run
python3 -m caramba run caramba/config/presets/llama32_1b_dba.yml --group quick

# Compile only (with optional plan output)
python3 -m caramba compile caramba/config/presets/llama32_1b_dba.yml --print-plan
```

> **Note:** MPS works out of the box on Apple Silicon. For CUDA, install a CUDA-enabled torch build separately.

## Manifests

A manifest declares:

| Section | What it defines |
|---------|-----------------|
| **Model** | Embedder + topology (the layer graph) |
| **Groups** | Collections of runs (e.g., upcycle → finetune) |
| **Runs** | Mode, seed, steps, training config |
| **Verify** | Post-run comparison tests (per run) |
| **Benchmarks** | Perplexity, latency, memory benchmarks (per group) |

Presets live in `caramba/config/presets/`. The compiler that lowers manifests into executable plans lives in `caramba/compiler/`.

---

## Upcycling: Llama → DBA

caramba can transplant **Decoupled Bottleneck Attention** into a pretrained Llama checkpoint, then train via blockwise distillation and global fine-tuning.

This workflow draws from [Attention Surgery](https://arxiv.org/abs/2509.24899) (Ghafoorian et al., 2025).

### The DBA Advantage

| Metric | Standard Attention | DBA (sem=128, geo=256) |
|--------|-------------------|------------------------|
| KV-Cache per token | 2048 bytes | 384 bytes |
| **Reduction** | 1x | **5.3x** |

DBA separates attention into:
- **Semantic path** (128 dims): Content/meaning, no positional encoding
- **Geometric path** (256 dims): Position, RoPE applied

### Run the Llama 3.2 1B → DBA upcycle

```bash
# 1. Prepare dataset (if needed)
python3 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy

# 2. Authenticate with HuggingFace (for gated models)
huggingface-cli login

# 3. Run full experiment with benchmarks
python3 -m caramba run caramba/config/presets/llama32_1b_dba.yml --group paper
```

The preset targets `mps` with `float32`. Edit the manifest to change device or dtype.

### What happens

1. **Build** teacher and student from manifest
2. **Load** teacher checkpoint, apply weights to both
3. **Surgery** — SVD-based initialization of DBA projections from teacher Q/K
4. **Blockwise distillation** — L1 loss on attention outputs, layer by layer
5. **Global fine-tuning** — cross-entropy on next-token prediction
6. **Benchmark** — perplexity, latency, memory comparison
7. **Artifacts** — CSV, JSON, PNG charts, LaTeX tables

---

## Verification

Attach a `verify:` block to any run to automatically compare teacher vs. student after training completes.

```yaml
runs:
  - id: blockwise
    mode: train
    # ... training config ...
    verify:
      type: compare
      batches: 2
      attention:
        max_mean_l1: 0.05   # Mean L1 across attention outputs
        max_max_l1: 0.25    # Max L1 (worst-case divergence)
      logits:
        max_mean_l1: 0.05   # Mean L1 across final logits
        max_max_l1: 0.25
```

Both models run on identical input batches. L1 agreement is measured at two levels: per-layer attention outputs `(B,T,D)` and full model logits `(B,T,V)`. Thresholds are enforced—exceed them and the run fails fast.

---

## Benchmarks

Groups can declare benchmark suites that run after all training completes and generate paper-ready artifacts.

```yaml
groups:
  - name: paper
    runs: [...]
    benchmarks:
      # Perplexity benchmark
      - id: perplexity
        config:
          type: perplexity
          dataset: "fineweb_100m.npy"
          block_size: 2048
          num_batches: 100
        models: ["teacher", "student"]

      # Latency benchmark
      - id: latency
        config:
          type: latency
          prompt_lengths: [128, 512, 1024, 2048]
          generation_lengths: [64, 128, 256]
          warmup_runs: 3
          timed_runs: 10
        models: ["teacher", "student"]

      # Memory benchmark
      - id: memory
        config:
          type: memory
          sequence_lengths: [512, 1024, 2048, 4096]
          quantization_modes: ["fp16", "q8", "q4"]
        models: ["teacher", "student"]
```

### Benchmark Types

| Type | Measures |
|------|----------|
| `perplexity` | Language modeling quality (cross-entropy loss) |
| `latency` | Tokens/second, prefill time, decode time |
| `memory` | KV-cache size, peak memory, quantization impact |
| `accuracy` | Task-specific metrics (HuggingFace evals) |
| `generation` | Text generation quality |

---

## Artifacts

After running with benchmarks, caramba generates publication-ready artifacts:

```
artifacts/llama32_1b_dba_upcycle_20241226_143000/
├── report.json              # Complete experiment metadata + summary
├── perplexity.csv           # Raw perplexity data
├── latency.csv              # Raw latency measurements
├── memory.csv               # Raw memory data
├── summary.png              # 3-panel comparison chart
├── latency_vs_context.png   # Throughput scaling visualization
├── memory_scaling.png       # Memory vs sequence length
└── tables.tex               # LaTeX tables for paper inclusion
```

### Sample LaTeX Output

```latex
\begin{table}[h]
\centering
\caption{DBA Upcycle Results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Teacher} & \textbf{Student (DBA)} & \textbf{Change} \\
\midrule
Perplexity $\downarrow$ & 8.42 & 8.59 & 1.02$\times$ \\
Throughput (tok/s) $\uparrow$ & 156 & 234 & 1.50$\times$ \\
KV-Cache (bytes/tok) $\downarrow$ & 2048 & 384 & 5.33$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## KV-Cache

caramba includes a production-grade KV-cache implementation with:

- **Quantization**: FP16, FP32, Q8, Q4, NF4
- **Decoupled caches**: Separate storage for semantic and geometric keys
- **Speculative decoding**: `truncate()` for rollback
- **Amortized allocation**: Geometric growth to avoid O(N) allocations

```python
from caramba.cache import LayerKVCache, DecoupledLayerKVCache
from caramba.config.kvcache import KVCacheTensorConfig, KVCacheKind

# Standard cache
cache = LayerKVCache(
    batch_size=1,
    max_seq_len=2048,
    k_dim=256,
    v_dim=256,
    k_cfg=KVCacheTensorConfig(kind=KVCacheKind.Q8),
    v_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16),
    device=torch.device("mps"),
)

# Decoupled cache for DBA
cache = DecoupledLayerKVCache(
    batch_size=1,
    max_seq_len=2048,
    k_sem_dim=128,  # Semantic keys
    k_geo_dim=256,  # Geometric keys
    v_dim=256,
    device=torch.device("mps"),
)
```

---

## Attention Layer

The unified `AttentionLayer` supports three modes:

| Mode | Description |
|------|-------------|
| `STANDARD` | Classic multi-head attention |
| `GQA` | Grouped-Query Attention (fewer KV heads) |
| `DECOUPLED` | DBA with semantic/geometric split |

```python
from caramba.layer.attention import AttentionLayer
from caramba.config.layer import AttentionLayerConfig, AttentionMode

# DBA attention
config = AttentionLayerConfig(
    d_model=2048,
    n_heads=32,
    n_kv_heads=8,
    mode=AttentionMode.DECOUPLED,
    sem_dim=128,
    geo_dim=256,
    rope_enabled=True,
    decoupled_gate=True,
)
layer = AttentionLayer(config)
```

---

## Architecture

<div align="center">

| | |
|:---:|:---:|
| [High-level overview](caramba/high-level.png) | [Detailed architecture](caramba/architecture.png) |
| [Llama-specific](caramba/llama-architecture.png) | [Preset manifests](caramba/config/presets/) |

</div>

---

## Testing

```bash
# Run all tests
python3 -m unittest discover -s caramba -p '*_test.py' -v

# Run specific test module
python3 -m unittest caramba.layer.attention_test -v
```

---

<div align="center">

*caramba* — because research shouldn't fight the tooling.

</div>
