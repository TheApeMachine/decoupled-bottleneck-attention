# Technical Specification: Optimizations

## Technical Context
- Language: Python (runtime observed: Python 3.9)
- Core dependencies: PyTorch; optional Triton for fused decode kernels
- Attention implementation of interest:
  - `production/attention_impl/decoupled_attention_impl/attention_core.py`
  - Triton kernels: `production/attention_impl/decoupled_attention_impl/kernels_q4q8q4.py`
- Tuning/config surface:
  - Minimal CLI: `production/cli.py`
  - Runtime sample runner: `production/runner_sample.py`
  - Run config parsing: `production/run_config.py`
  - Self-opt config: `production/optimizer/tuner/config.py` (`KVSelfOptConfig`)
- Tests: `make test` (unittest discovery)

## Primary Constraints (Must Hold)
1. **Decoupled additive-logit invariant**
   - Semantic and geometric logits must be defined over the same token positions so they can be added.
   - “N→M compress only semantic K/V while leaving geometric full length” is not allowed.
2. **Causality / cache-compatibility**
   - No train/infer mismatch from conditioning on future tokens.
   - No mechanism that invalidates or requires recomputing existing KV cache entries.
3. **Null token fused decode correctness**
   - Null token must be included exactly once globally.
   - No per-partition duplication in 2-pass (partitioned) fused decode.
4. **Zero-penalty when null disabled**
   - Null-token work must be compiled out of Triton kernels via a `tl.constexpr` flag.
5. **Codebase rules**
   - Imports at top of Python files only.
   - No `Any`; keep typing correct.

## Decisions (From User)
- `production/cli.py` will expose `--kv-policy` as an advanced escape hatch while keeping CLI otherwise minimal.
- Long-seq memory summarizer default: implement the learned summarizer and make it the effective default in long-seq mode, but initialize it to match the current mean summarizer to minimize drift.
- New input-conditioned gating default: enabled by default (with neutral initialization to avoid abrupt behavior change).
- Fused-vs-streaming parity tolerance: aim for the tightest workable tolerance; implement a strict default and include clear diagnostics.

## Implementation Approach

### A) Correctness Cleanup
**Bug fix:** remove the duplicate `out_proj` call in the decoupled null-attn full-attention path.

### B) Long-Sequence Approximation: Learned Memory Summarizer
The existing training-time long-seq approximation constructs:
- semantic-only “memory tokens” summarizing older context
- plus a local full-resolution tail window for both semantic and geometric paths

We extend only the memory-token builder.

**New config knobs (explicit, not `getattr`-hidden):**
- `train_long_seq_enabled: bool`
- `train_long_seq_threshold: int | None`
- `train_long_seq_mem_block: int | None`
- `train_long_seq_local_window: int | None`
- `train_long_seq_q_chunk: int | None`
- `train_long_seq_mem_summarizer: Literal["mean", "linear", "conv"]`

**Source of truth fix:**
- Ensure the resolved model/attention config includes these fields explicitly.
- Remove implicit dynamic attribute injection assumptions.

**Summarizer behavior:**
- `mean`: existing behavior.
- `linear`: learned pooling per block producing one token per block.
- `conv`: strided/depthwise conv summarizer producing one token per block.

**Default choice and stability:**
- Default `train_long_seq_mem_summarizer="conv"` on CUDA and `"linear"` otherwise (or `"mean"` on backends where conv is unsupported).
- Initialize learned summarizers so that the initial output exactly matches the mean summarizer (or matches closely within fp precision):
  - Use an explicit “mean + residual” parameterization with residual scale initialized to 0, or initialize convolution/linear weights to compute a mean.

**Integration point:**
- Implement summarizer as a small module owned by the decoupled attention module (so it is checkpointed), used only in the long-seq branch.

### C) Causal Input-Conditioned Gating (Default On)
Replace the purely static learned per-head sem/geo gate with an input-conditioned gate computed per token, designed to be cache-compatible.

**Design:**
- Compute `g(x_t)` from the current token hidden state (or other strictly causal features) producing a per-head gate `g ∈ [0,1]^H`.
- Apply as query scaling, consistent with existing fused-kernel compatibility:
  - `q_sem *= 2*g`
  - `q_geo *= 2*(1-g)`

**Initialization:**
- Initialize gating to be neutral (outputs `g=0.5`) so effective scaling is ~1.0 at start.

**Optional extension (post-attention SE-style output gate):**
- Per-token vector gate over `v_head_dim` applied to the attention output before `out_proj`.
- Must be strictly causal.

### D) Null Token Integration Into Triton Fused Decode

#### D1) Kernel API
Update `kv_decode_update_decoupled_q4q8q4` and `kv_decode_partition_stats_decoupled_q4q8q4` (and reduction kernel as needed) to accept:
- `k_sem_null_ptr`, `k_geo_null_ptr`, `v_null_ptr`
- `HAS_NULL: tl.constexpr`

Null tensors layout:
- Prefer contiguous fp16/fp32 buffers of shape `(B, H, HD_*)` (consistent with how q vectors are currently passed).
- Pass explicit strides for null pointers (or enforce contiguous and document).

#### D2) Correct math + seeding
Compute null logit using the same scaling as the main logits:
- `s_null = dot(q_sem, k_sem_null) * SEM_SCALE + dot(q_geo, k_geo_null) * GEO_SCALE`

Seed online-softmax state before any KV iteration:
- `m = s_null`
- `d = 1`
- `o = v_null` (fp32 accumulation)

#### D3) 1-pass kernel
In `kv_decode_update_decoupled_q4q8q4`, if `HAS_NULL`:
- Load null vectors once per program
- Seed `(m,d,o)`
- Continue with the existing KV loop

#### D4) 2-pass kernel (partition + reduce)
Avoid duplicating null per partition.

Preferred design:
- Do not seed null in `kv_decode_partition_stats_decoupled_q4q8q4`.
- Extend `kv_decode_reduce_partitions` to incorporate the null token once by treating it as a “virtual partition” during reduction:
  - Compute global max `m = max(max_p m_part[p], s_null)`
  - Compute `d = sum_p d_part[p] * exp(m_part[p]-m) + 1 * exp(s_null - m)`
  - Compute `o = sum_p o_part[p] * exp(m_part[p]-m) + v_null * exp(s_null - m)`

This keeps correctness independent of partitioning strategy.

#### D5) Python plumbing
In `attention_core.py`:
- Remove the `RuntimeError` guards that forbid fused decode under `null_attn`.
- Pass null tensors into the 1-pass and 2-pass kernel launches.
- Update fused eligibility (`fused_ok`) to allow null-attn only when the null-enabled kernels are available.
- Preserve fallback: any fused failure falls back to streaming decode.

### E) KV Policy UX: Minimal CLI + `--kv-policy`
Expose only a single advanced flag in `production/cli.py`:
- `--kv-policy` (string)

Behavior:
- If provided, parse with `KVCachePolicy.parse()` and treat as atomic override over all granular KV flags.
- Detect legacy `--kv-cache-*`/related flags via improved `_MinimalParser.error()` messaging:
  - If user passes deprecated flags, print: “Use --kv-policy for a unified configuration.”

Maintain intent-first UX:
- Do not expose the granular KV flags in the minimal CLI.

### F) Run Summary Updates
In `production/runner_train_impl/summary.py`, extend summary table:
- `KV Policy`: short-string of the active policy (if known)
- `Null Attention`: `Inactive` / `Active (Unfused)` / `Active (Fused)`

## Source Code Structure Changes
Expected edits (non-exhaustive):
- `production/attention_impl/decoupled_attention_impl/attention_core.py`
  - fix `out_proj` double-application
  - learned memory summarizer module + explicit config plumbing
  - input-conditioned gating (default on)
  - fused decode null-token plumbing
- `production/attention_impl/decoupled_attention_impl/kernels_q4q8q4.py`
  - add null pointers + `HAS_NULL`
  - correct 1-pass seeding
  - integrate null in 2-pass reduce kernel
- `production/cli.py`
  - add `--kv-policy` only (advanced)
  - improved deprecated flag hint
- `production/run_config.py` / `production/runner_sample.py`
  - ensure CLI-provided `kv_policy` is correctly respected as atomic override
- `production/runner_train_impl/summary.py`
  - add KV policy + null attention state rows

New scripts/tests:
- Add a CUDA-only parity test for fused decode with null enabled.
- Add a long-seq drift validator aligned with existing metric style.

## Interfaces / Contracts

### Kernel call contract (Python → Triton)
Add optional null args and a `HAS_NULL` constexpr. Kernel launch must:
- pass valid pointers (or dummy pointers) consistent with `HAS_NULL`
- ensure dtype and layout match kernel expectations

### Config contract
- The attention module must receive long-seq knobs from a single explicit configuration object/fields.
- `--kv-policy` must override any derived/default KV cache configuration in sampling.

## Delivery Phases (Incremental Milestones)
1. Fix `out_proj` duplication bug; add a unit test that would have caught it.
2. Implement `--kv-policy` in minimal CLI + error hinting; add parsing tests.
3. Add input-conditioned gating (default on) with neutral initialization; add a correctness/regression test for shape/dtype.
4. Implement learned memory summarizer with mean-matching initialization; add a long-seq unit test that validates shape invariants and fallback.
5. Implement Triton 1-pass null-token integration + parity test vs streaming decode.
6. Implement Triton 2-pass null-token integration in reduce + parity test vs streaming decode.
7. Update run summary rows and validate output.
8. Add long-seq drift validator script aligned with existing metrics.

## Verification
Local:
- `make test`

GPU-specific (when CUDA + Triton available):
- Run parity tests for:
  - fused 1-pass decode (null enabled) vs streaming
  - fused 2-pass decode (null enabled) vs streaming

Tolerance policy (tight default):
- Default compare: `max_abs <= 1e-3` and `max_rel <= 1e-3` for fp16.
- If bf16 is used, allow a slightly looser threshold (documented) only if necessary.
- Tests must print diagnostics (max_abs, max_rel, dtype, sizes) on failure.

