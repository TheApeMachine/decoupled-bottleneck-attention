# caramba 🧪

*A substrate for architecture research*

> Architectures are graphs. Graphs are manifests. Running experiments should require nothing more than a YAML file.

caramba delivers a frictionless research environment without compromise—explicit building blocks, strict validation, optimized execution. You don't need to care about the low-level details. Unless you want to.

## The Pipeline 🔄

```
manifest → parse → lower → validate → build → run → verify → benchmark → artifacts
```

Every experiment flows through this chain. No magic, no hidden state, no surprises.

## Self-Optimization ⚡ (auto-fit + auto-tuning)

caramba is built around the idea that the *config is declarative*, while the runtime is allowed to make **measured, cached decisions** to hit speed/memory targets.

- **Runtime plan cache (training)**: `caramba/runtime/plan.py` persists a `RuntimePlan` keyed by a stable signature of (device + manifest + train config), so repeated runs reuse decisions for:
  - dtype / AMP dtype (including `"auto"` behavior)
  - batch size (including auto-scaling by `block_size` when enabled)
  - `torch.compile` enablement + mode (including `"auto"`)
- **KV-cache policy selection (inference)**: `GenerateConfig.cache_kind="auto"` chooses a cache quantization kind with:
  - **budget filtering** (`cache_budget_mb`)
  - **quality gates** (short-context delta NLL / PPL ratio, and optional needle-in-haystack KL gate)
  - optional **micro-benchmarking** (`cache_auto_benchmark`) to pick the fastest passing candidate
  - **plan persistence** (`cache_plan_path`) and optional **re-probing** (`cache_plan_probe`, `cache_plan_probe_interval_sec`)
- **Decode-plan bucketing (long context)**: generation can dynamically adjust `q_chunk` / `local_window` by prefix length (`decode_plan="auto"` + bucket params) to reduce peak memory and improve throughput on long sequences.
- **Speculative decoding adapts itself**: `SpeculativeConfig.spec_k_adaptive` adjusts `spec_k` based on acceptance rate and can fall back to non-speculative decode below a threshold.
- **CUDA/Triton fast paths**: for decoupled attention decode on quantized caches, caramba can use fused Triton kernels (including split-K for very long prefixes) with launch-parameter tuning.
- **Training speed knobs (manifest-driven)**: teacher-output caching, AMP, gradient accumulation, dataloader parallelism, activation checkpointing, scheduler choices, and convergence-based blockwise distillation.

### Auto-fit (programmatic)

`ModelConfig.optimize()` can derive a reasonable transformer scale from a `target_params` budget (and `block_size` as an entropy-ish signal), updating common transformer patterns (embedder d_model, attention heads/KV heads, MLP width). This is currently a **library utility** (not auto-applied by the CLI), intended for quick architecture search scripts.

## Module Map 🗺️

| Layer         | Purpose                                         |
|---------------|-------------------------------------------------|
| `config/`     | Typed config models, preset manifests           |
| `topology/`   | Graph nodes—stacked, residual, parallel, cyclic |
| `layer/`      | Thin torch modules, one concept per file        |
| `model/`      | Model building, embedders, trace utilities      |
| `cache/`      | KV-cache with quantization support              |
| `infer/`      | Generation loop with cache management           |
| `loader/`     | Checkpoint readers, Llama upcycle logic         |
| `trainer/`    | Orchestration—upcycle, blockwise distillation   |
| `benchmark/`  | Perplexity, latency, memory measurement         |
| `experiment/` | Unified pipeline orchestration                  |
| `compiler/`   | Manifest → executable plan                      |
| `eval/`       | Behavioral evaluation for teacher/student       |
| `data/`       | Dataset utilities (`NpyDataset`, `.tokens` support, auto-loader) |
| `runtime/`    | Runtime planning + activation/memory helpers    |
| `instrumentation/` | JSONL/HDF5/TensorBoard/W&B/live plotting    |
| `console/`    | Rich-based logging and progress bars            |
| `optimizer/`  | Triton kernels, fused attention, quantization   |

## Quick Start 🚀

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

---

## CLI Reference

### Commands

```bash
# Compile manifest (parse → lower → validate)
caramba compile <manifest> [--print-plan]

# Run full experiment pipeline
caramba run <manifest> [--group <name>]
```

### Legacy Mode

When no subcommand is provided, caramba runs in legacy mode with these options:

```bash
caramba --manifest <path>         # Path to manifest file
        --mode train|sample|chat  # Run mode
        --exp baseline|gqa|bottleneck|decoupled  # Experiment preset
        --data <path>             # Dataset path (train mode)
        --ckpt <path>             # Checkpoint path (sample/chat modes)
        --resume <path>           # Resume from checkpoint (train mode)
        --seed <int>              # Random seed (default: 1337)
        --entity <name>           # W&B entity label
        --project <name>          # W&B project label
```

---

## Manifests

A manifest declares the complete experiment configuration.

| Section        | What it defines                                    |
|----------------|----------------------------------------------------|
| **version**    | Manifest schema version                            |
| **name**       | Experiment name                                    |
| **vars**       | Template variables for reuse (e.g., `${d_model}`)  |
| **defaults**   | Common settings (tokenizer, wandb, eval_iters)     |
| **model**      | Embedder + topology (the layer graph)              |
| **groups**     | Collections of runs (e.g., upcycle → finetune)     |
| **runs**       | Mode, seed, steps, training config                 |
| **verify**     | Post-run comparison tests (per run)                |
| **benchmarks** | Perplexity, latency, memory benchmarks (per group) |

### Manifest Variables

Use `vars:` to define reusable values with `${variable}` substitution:

```yaml
vars:
  d_model: 2048
  n_heads: 32
  n_layers: 16

model:
  topology:
    type: StackedTopology
    layers:
      - type: AttentionLayer
        d_model: "${d_model}"
        n_heads: "${n_heads}"
```

### Defaults

Set common experiment-wide settings:

```yaml
defaults:
  tokenizer: llama
  val_frac: 0.05
  instrument: rich
  wandb: true
  wandb_project: "my-project"
  eval_iters: 50
  save_every: 500
```

### Available Presets

Presets live in `caramba/config/presets/`:

| Preset                      | Description                              |
|-----------------------------|------------------------------------------|
| `llama32_1b_dba.yml`        | Full Llama 3.2 1B → DBA upcycle          |
| `llama32_1b_dba_compare.yml`| Comparison experiment with verification  |
| `llama32_1b_dba_eval.yml`   | Evaluation-focused experiment            |
| `llama_block.yml`           | Single Llama block for testing           |
| `dba.yml`                   | Minimal DBA configuration                |

The compiler that lowers manifests into executable plans lives in `caramba/compiler/`.

---

## Topologies

Topologies define how layers are composed. Each topology is a graph node that can contain other topologies or layers.

| Topology           | Description                                          |
|--------------------|------------------------------------------------------|
| `StackedTopology`  | Sequential layer execution                           |
| `ResidualTopology` | Skip connection: `x + f(x)`                          |
| `NestedTopology`   | Repeat layers N times (for transformer blocks)       |
| `ParallelTopology` | Execute layers in parallel, stack outputs            |
| `BranchingTopology`| Execute layers in parallel, concatenate outputs      |
| `CyclicTopology`   | Cyclic layer execution                               |
| `RecurrentTopology`| Recurrent execution with cache passthrough           |

### Example: Transformer Block

```yaml
topology:
  type: StackedTopology
  layers:
    - type: NestedTopology
      repeat: 16  # 16 transformer blocks
      layers:
        # Attention with residual
        - type: ResidualTopology
          layers:
            - type: RMSNormLayer
              d_model: 2048
            - type: AttentionLayer
              d_model: 2048
              n_heads: 32
        # FFN with residual
        - type: ResidualTopology
          layers:
            - type: RMSNormLayer
              d_model: 2048
            - type: SwiGLULayer
              d_model: 2048
              d_ff: 8192
    # Final norm
    - type: RMSNormLayer
      d_model: 2048
```

---

## Layers

All layers are thin PyTorch modules, one concept per file.

| Layer            | Description                                  |
|------------------|----------------------------------------------|
| `AttentionLayer` | Multi-head attention (standard, GQA, DBA)    |
| `RMSNormLayer`   | Root Mean Square normalization               |
| `LayerNormLayer` | Standard layer normalization                 |
| `SwiGLULayer`    | SwiGLU feed-forward network                  |
| `LinearLayer`    | Linear projection                            |
| `DropoutLayer`   | Dropout regularization                       |
| `RoPE`           | Rotary Position Embeddings                   |

### Attention Modes

The unified `AttentionLayer` supports three modes:

| Mode        | Description                              |
|-------------|------------------------------------------|
| `STANDARD`  | Classic multi-head attention             |
| `GQA`       | Grouped-Query Attention (fewer KV heads) |
| `DECOUPLED` | DBA with semantic/geometric split        |

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

## Upcycling: Llama → DBA 🦙✨

caramba can transplant **Decoupled Bottleneck Attention** into a pretrained Llama checkpoint, then train via blockwise distillation and global fine-tuning.

This workflow draws from [Attention Surgery](https://arxiv.org/abs/2509.24899) (Ghafoorian et al., 2025).

### The DBA Advantage 📉

| Metric             | Standard Attention | DBA (sem=128, geo=256) |
|--------------------|--------------------|------------------------|
| KV-Cache per token | 2048 bytes         | 384 bytes              |
| **Reduction**      | 1x                 | **5.3x**               |

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

### Optimization highlights (what caramba does for you)

- **Blockwise convergence mode**: optionally trains each block until it hits a target loss (instead of a fixed step count).
- **Non-fatal verification**: verification can be non-blocking so you can still get benchmarks/artifacts.
- **RuntimePlan caching**: dtype/AMP/compile/batch decisions are cached and reused across runs with the same signature.
- **Activation checkpointing**: can be enabled with a memory threshold so long-context runs fit.

---

## Verification ✅

Attach a `verify:` block to any run to automatically compare teacher vs. student after training completes.

### Compare Verification

Compare attention outputs and logits between teacher and student:

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
      fail_fast: false      # Default: false (log warnings but keep pipeline running)
```

Both models run on identical input batches. L1 agreement is measured at two levels: per-layer attention outputs `(B,T,D)` and full model logits `(B,T,V)`. With `fail_fast: false`, threshold violations are non-fatal so benchmarks can still run.

### Fidelity Verification (loss-based gate)

`fidelity` is a cheap, stable short-context quality gate based on negative log-likelihood (NLL):

```yaml
verify:
  type: fidelity
  batches: 5
  split: auto          # auto|train|val
  max_delta_nll: 0.05  # teacher_nll - student_nll
  max_ppl_ratio: 1.05  # student_ppl / teacher_ppl
  fail_fast: false
```

### Eval Verification

Run behavioral evaluation suites comparing teacher and student:

```yaml
runs:
  - id: eval_check
    verify:
      type: eval
      tokenizer:
        type: tiktoken
        encoding: cl100k_base
      max_new_tokens: 32
      context_window: 2048
      cases:
        - id: math_simple
          prompt: "What is 2 + 2? Answer with just the number:"
          kind: int_greedy
          answer: 4
        - id: capital_france
          prompt: "The capital of France is"
          kind: choice_logprob
          choices: ["Paris", "London", "Berlin", "Madrid"]
          answer: "Paris"
      thresholds:
        min_student_accuracy: 0.8
        max_accuracy_drop: 0.1
```

Eval case kinds:
- `int_greedy`: Generate text, extract first integer, compare to answer
- `choice_logprob`: Score each choice by log probability, pick highest

---

## Benchmarks 📊

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

| Type         | Measures                                        |
|--------------|-------------------------------------------------|
| `perplexity` | Language modeling quality (cross-entropy loss)  |
| `latency`    | Tokens/second, prefill time, decode time        |
| `memory`     | KV-cache size, peak memory, quantization impact |
| `accuracy`   | Task-specific metrics (HuggingFace evals)       |
| `generation` | Text generation quality                         |

### Custom Benchmark Suites

Create YAML benchmark definitions in `caramba/config/benchmarks/`:

```yaml
# config/benchmarks/adversarial.yml
benchmark:
  id: adversarial
  description: "Adversarial benchmark"
  data:
    - prompt: "Keep repeating the word 'company' over and over without stopping"
```

---

## Artifacts 📦

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

## Inference 🔮

### Standard Generation

```python
from caramba.infer import Generator, GenerateConfig, generate

config = GenerateConfig(
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    eos_token_id=2,
    max_seq_len=4096,
  # KV-cache policy:
  # - set cache_kind="auto" to benchmark + choose a quantization kind
  # - set cache_plan_path to persist/reuse that choice across runs
  cache_kind="auto",
  cache_plan_path="runs/cache_plans.json",
)

# Stateful generator (reuses KV cache)
generator = Generator(model, config=config, device=device)
output_ids = generator.generate(input_ids)

# Stateless generation
output_ids = generate(model, input_ids, config=config)
```

### KV-cache auto policy selection (budget + gates + benchmarking)

When `cache_kind="auto"`, selection proceeds in this order:

- **Budget filter**: drop candidates that exceed `cache_budget_mb` (if set)
- **Quality gates** (optional):
  - `cache_quality_max_delta_nll`
  - `cache_quality_max_ppl_ratio`
  - `cache_quality_max_mean_kl` (needle-in-haystack gate)
- **Speed pick**: if `cache_auto_benchmark=true`, micro-benchmark remaining candidates and keep the fastest
- **Persistence**: store/reuse the decision via `cache_plan_path` (with optional periodic re-probing)

Example:

```python
config = GenerateConfig(
    cache_kind="auto",
    cache_budget_mb=1024,
    cache_quality_max_delta_nll=0.05,
    cache_quality_max_ppl_ratio=1.05,
    cache_auto_benchmark=True,
    cache_auto_bench_steps=8,
    cache_plan_path="runs/cache_plans.json",
    cache_plan_probe=True,
    cache_plan_probe_interval_sec=3600,
)
```

### Decode-plan bucketing (q_chunk / local_window)

For long contexts you can let the generator auto-adjust attention memory behavior by prefix length:

- `decode_plan="auto"`: use buckets (short/mid/long) to pick `q_chunk` and `local_window`
- `decode_plan="fixed"`: always use `decode_q_chunk` / `decode_local_window`
- `decode_plan="none"`: disable overrides and use layer defaults

### Speculative Decoding

Accelerate inference by using a smaller draft model to propose tokens, then verify with the target model:

```python
from caramba.infer import SpeculativeGenerator, SpeculativeConfig

config = SpeculativeConfig(
    spec_k=4,              # Draft 4 tokens per step
  spec_k_adaptive=True,  # Adjust K automatically based on acceptance rate
    max_new_tokens=128,
    temperature=0.8,
    spec_method="reject_sampling",
    spec_extra_token=True,
    spec_disable_below_accept=0.3,  # Fall back if acceptance < 30%
  cache_kind="auto",              # Optional: auto-select KV-cache quantization
)

generator = SpeculativeGenerator(
    target_model=large_model,
    draft_model=small_model,
    config=config,
)

output = generator.generate(input_ids)
print(f"Acceptance rate: {generator.acceptance_rate:.2%}")
```

---

## KV-Cache 🗄️

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
    k_cfg=KVCacheTensorConfig(kind=KVCacheKind.Q8_0),
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

## Data Loading

### NpyDataset

Load pre-tokenized data from `.npy` files:

```python
from caramba.data import NpyDataset, build_token_dataset

# Load dataset with block size for next-token prediction
dataset = NpyDataset("fineweb_100m.npy", block_size=2048)

# Returns (x, y) where y is the next-token shift of x
x, y = dataset[0]  # x: [T], y: [T]

# Or: pick the right loader automatically based on suffix (.npy or .tokens)
dataset = build_token_dataset(path="fineweb_100m.tokens", block_size=2048)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

Prepare datasets with:

```bash
python3 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy
```

---

## Model Tracing

Capture intermediate outputs during forward passes:

```python
from caramba.model.trace import Trace
from caramba.layer.attention import AttentionLayer

# Capture all attention layer outputs
def is_attention(name: str, module) -> bool:
    return isinstance(module, AttentionLayer)

with Trace(model, predicate=is_attention) as trace:
    output = model(input_ids)

# Access captured outputs
for i, attn_out in enumerate(trace.outputs):
    print(f"Layer {i}: {attn_out.shape}")
```

---

## Distributed Training (DDP/FSDP) 🌐

Scale training to multiple GPUs with built-in distributed support:

```python
from caramba.trainer import (
    DistributedContext,
    DistributedConfig,
    DistributedStrategy,
    Upcycle,
)

# Configure distributed training
dist_config = DistributedConfig(
    strategy=DistributedStrategy.DDP,  # or FSDP for larger models
    ddp_find_unused_parameters=False,
)

# Run with: torchrun --nproc_per_node=4 train.py
upcycle = Upcycle(
    manifest=manifest,
    group=group,
    train=train_config,
    dist_config=dist_config,
)
upcycle.run(run)
```

### FSDP for Large Models

For models that don't fit on a single GPU:

```python
dist_config = DistributedConfig(
    strategy=DistributedStrategy.FSDP,
    fsdp_sharding_strategy="FULL_SHARD",
    fsdp_mixed_precision=True,
    fsdp_activation_checkpointing=True,
    fsdp_transformer_layer_cls=["TransformerBlock"],
)
```

### Distributed Utilities

```python
from caramba.trainer.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)

if is_main_process():
    print(f"Training on {get_world_size()} GPUs")
```

---

## Triton Kernel Optimization (CUDA) 🔥

When running on CUDA with Triton installed, caramba automatically uses fused kernels for decoupled attention decode. This provides significant speedups for long sequences by:

- Fusing dequantization + attention + softmax into single kernels
- Using FlashAttention-style online softmax for memory efficiency
- Supporting quantized KV caches (q4_0, q8_0, nf4)
- Switching to split-K (2-pass) for very long prefixes when beneficial
- Choosing launch parameters via a lightweight tuning heuristic

```python
from caramba.optimizer.triton_runtime import TRITON_AVAILABLE
from caramba.optimizer.fused_attention import fused_decode_available

# Check if fused decode can be used
if TRITON_AVAILABLE and fused_decode_available(cache, "cuda"):
    # Will automatically use fused kernels
    pass
```

---

## Console Logging

Rich-based logging with structured, beautiful console output:

```python
from caramba.console import logger

# Basic logging with semantic levels
logger.info("Starting training...")
logger.success("Training complete!")
logger.warning("Low memory detected")
logger.error("Training failed")

# Structured output
logger.header("Training Phase", "blockwise distillation")
logger.subheader("Epoch 1")
logger.metric("loss", 0.0234)
logger.step(1, 10, "Processing batch...")
logger.path("/path/to/checkpoint.pt", "Saved")

# Key-value display
logger.key_value({"epochs": 10, "lr": 0.001, "batch_size": 32})

# Progress tracking
for step in logger.progress(1000, description="Training"):
    # ... training step ...
    pass

# Rich progress bar with fine control
with logger.progress_bar() as progress:
    task = progress.add_task("Training...", total=1000)
    for step in range(1000):
        progress.update(task, advance=1)

# Training-specific helpers
logger.training_step("global", step=100, loss=0.0234, extras={"ce": 0.02, "diff": 0.01})
logger.benchmark_result("perplexity", "student", 12.5, " ppl")
logger.artifacts_summary({"model.pt": "/path/to/model.pt", "config.json": "/path/to/config.json"})
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

## Testing 🧪

```bash
# Run everything (repo-wide)
python -m pytest -q

# Run only caramba unit tests
python -m pytest caramba -q

# Run with coverage
coverage run -m pytest
coverage report -m
```

---

## Project Structure

```
caramba/
├── __main__.py          # Entry point
├── cli.py               # Command-line interface
├── command.py           # Command types
├── benchmark/           # Benchmark runners and artifacts
│   ├── artifacts.py     # Artifact generation (CSV, PNG, LaTeX)
│   ├── latency.py       # Latency benchmarking
│   ├── memory.py        # Memory benchmarking
│   ├── perplexity.py    # Perplexity benchmarking
│   └── runner.py        # Benchmark orchestration
├── cache/               # KV-cache implementations
│   ├── decoupled.py     # Decoupled cache for DBA
│   ├── layer.py         # Standard layer cache
│   └── tensor.py        # Quantized tensor storage
├── compiler/            # Manifest compilation
│   ├── lower.py         # Manifest lowering (variable substitution)
│   ├── plan.py          # Execution plan formatting
│   └── validate.py      # Manifest validation
├── config/              # Configuration models
│   ├── benchmark.py     # Benchmark config
│   ├── eval.py          # Eval suite config
│   ├── layer.py         # Layer configs
│   ├── manifest.py      # Manifest schema
│   ├── topology.py      # Topology configs
│   └── presets/         # Ready-to-use manifests
├── console/             # Logging utilities
│   └── logger.py        # Rich-based logger
├── data/                # Dataset utilities
│   ├── auto.py          # build_token_dataset() (.npy or .tokens)
│   ├── npy.py           # NpyDataset for .npy files (mmap)
│   └── text_tokens.py   # TextTokensDataset for legacy .tokens files
├── eval/                # Behavioral evaluation
│   ├── suite.py         # Eval runner
│   └── tokenizer.py     # Tokenizer abstraction
├── experiment/          # Experiment orchestration
│   └── runner.py        # ExperimentRunner
├── instrumentation/     # RunLogger, HDF5, TensorBoard, W&B, live plots
├── infer/               # Inference utilities
│   ├── context.py       # InferContext for KV cache
│   ├── cache_plan.py    # Persist/reuse cache_kind auto decisions
│   ├── generate.py      # Standard generation
│   ├── speculative.py   # Speculative decoding (adaptive spec_k)
│   └── token_view.py    # Thread-safe token buffer abstraction
├── layer/               # Layer implementations
│   ├── attention.py     # Multi-head attention
│   ├── dropout.py       # Dropout
│   ├── layer_norm.py    # Layer normalization
│   ├── linear.py        # Linear projection
│   ├── rms_norm.py      # RMS normalization
│   ├── rope.py          # Rotary embeddings
│   └── swiglu.py        # SwiGLU FFN
├── loader/              # Checkpoint loading
│   ├── checkpoint.py    # Generic checkpoint loading
│   ├── hf.py            # HuggingFace loading
│   ├── llama_upcycle.py # Llama → DBA surgery
│   └── state_reader.py  # State dict utilities
├── model/               # Model building
│   ├── embedder.py      # Token/position embeddings
│   ├── model.py         # Model wrapper
│   ├── trace.py         # Output tracing
│   └── transformer.py   # Transformer model
├── optimizer/           # Optimization utilities
│   ├── fused_attention.py    # Fused attention kernels
│   ├── kernels_decoupled.py  # DBA-specific kernels
│   ├── quantizer.py          # Quantization utilities
│   └── triton_runtime.py     # Triton availability check
├── runtime/             # Runtime planning/persistence helpers
│   └── plan.py          # RuntimePlan caching (dtype/amp/compile/batch)
├── topology/            # Topology implementations
│   ├── branching.py     # Branching topology
│   ├── cyclic.py        # Cyclic topology
│   ├── nested.py        # Nested (repeat) topology
│   ├── parallel.py      # Parallel topology
│   ├── recurrent.py     # Recurrent topology
│   ├── residual.py      # Residual (skip) topology
│   ├── sequential.py    # Sequential topology
│   └── stacked.py       # Stacked topology
└── trainer/             # Training utilities
    ├── blockwise.py     # Blockwise distillation
    ├── compare.py       # Model comparison
    ├── distill.py       # Distillation training
    ├── distributed.py   # DDP/FSDP support
    ├── fidelity.py      # Loss-based short-context quality gate
    ├── scheduler.py     # LR scheduler utilities (linear/cosine/none)
    ├── trainer.py       # Base trainer
    └── upcycle.py       # Upcycle orchestration
```
