# Product Requirements Document (PRD): Optimizations

## Summary
This work upgrades the decoupled (semantic + geometric additive-logit) attention stack with production-safe optimizations and removes the current “Null Token disables Triton fused decode” limitation.

The deliverables are explicitly constrained to remain compatible with the current decoupled design:
- Decoupled attention logits remain additive over the same token positions.
- Training-time long-sequence approximation remains a single-attention-call construction via “memory tokens + local tokens”, not an incompatible N→M projection that would break logit alignment.
- Any new adaptivity must be strictly causal and KV-cache compatible.

## Background / Current State
- `production/attention_impl/decoupled_attention_impl/attention_core.py` implements decoupled attention for training/full-attn, streaming decode, and optional Triton fused decode for quantized caches.
- Null-attention exists for manual and streaming decode but is hard-disabled for fused decode.
- A training-time long-sequence approximation already exists for decoupled mode: semantic “memory” summaries + full-res local window.
- Known bug: the decoupled null-attn full-attention path calls `out_proj` twice.

## Goals
1. Enable Null Token in Triton fused decode with **zero penalty when disabled**.
2. Fix correctness issues and reduce configuration ambiguity (clear source of truth for long-seq knobs).
3. Improve long-sequence training approximation by allowing a **learned memory summarizer** (still shape-compatible with additive logits).
4. Add **strictly causal**, cache-compatible adaptivity for sem/geo mixture (and optionally output gating) without train/infer mismatch.
5. Improve user visibility into the active KV policy and whether null-attention is fused.
6. Improve KV policy UX without compromising the project’s intent-first CLI philosophy.

## Non-Goals
- Do not implement per-sequence hypernetwork-generated Q/K projection weights that depend on future tokens.
- Do not introduce an architecture that produces semantic logits on a different token grid than geometric logits.
- Do not add a full adversarial attention-map gradient attack benchmark unless the repository already has a supported attention-map extraction + differentiable path.

## Users / Use Cases
- **Production inference user**: wants fast decoding with quantized caches and optional null-attention without losing fused performance.
- **Researcher/trainer**: wants long-seq training to scale better with minimal drift and with the option to learn better semantic memory summaries.
- **Operator/experimenter**: wants a single KV policy string that is easy to copy/paste, log, and reproduce.

## Requirements

### R0: Correctness / Cleanup
- Fix the double `out_proj` application in decoupled null-attn full-attention.
- Ensure that changes do not alter default behavior unless explicitly enabled/configured.

### R1: Long-Sequence Configuration Source of Truth
- Remove implicit reliance on `getattr(cfg, "train_long_seq_*", ...)` when the authoritative fields live elsewhere.
- Provide a single explicit path from “tuner/selfopt config” → “model/attention config” for long-seq approximation knobs.
- Document and log the resolved long-seq knobs used during training when enabled.

### R2: Learned Memory Summarizer (Training-Time Long-Seq Approximation)
Applies only to the existing long-seq approximation mode.

- Add a selectable memory summarizer for semantic memory tokens over fixed-size blocks.
- Supported modes (minimum):
  - `mean` (current behavior; baseline)
  - `conv` (learned strided/depthwise 1D conv summarizer)
  - `linear` (learned pooling/projection summarizer)
- The summarizer must output the same tensor shapes as the current memory-token builder so the subsequent single SDPA call remains valid.
- Geometric path contribution for memory tokens remains zero (semantic-only memory) to preserve additive-logit invariants.
- Must preserve correctness/compatibility conditions of the existing approximation:
  - Enabled only in training mode (unless explicitly expanded later)
  - Disabled when `attn_mask` is provided
  - Safe fallback to exact SDPA on any runtime error

### R3: Causal, Cache-Compatible Adaptivity (Replace “Dynamic Q/K HyperNetwork”)
- Provide an optional, strictly causal sem/geo mixture gate computed per token.
- The gate must be usable in:
  - full-attention (training)
  - streaming decode
  - Triton fused decode
- The gate must not require recomputing historical K/V and must not introduce train/infer mismatch.
- The preferred integration point is query scaling (consistent with existing per-head gate behavior).

Optional extension:
- Provide an optional per-channel output gate (SE-style) applied to attention outputs before `out_proj`, computed per token from causal features.

### R4: Null Token Integrated Into Triton Fused Decode (Zero-Penalty When Disabled)
Applies to quantized decoupled fused decode kernels.

- Add optional kernel arguments for null token K/V:
  - `k_sem_null_ptr`, `k_geo_null_ptr`, `v_null_ptr`.
- Implement `HAS_NULL: tl.constexpr` so the null-token logic is compiled out when disabled.
- Correct null-token logit math must match the existing fused logits:
  - `s_null = dot(q_sem, k_sem_null) * SEM_SCALE + dot(q_geo, k_geo_null) * GEO_SCALE`.
- Seed the online softmax state exactly once with the null token before iterating the prefix KV tokens:
  - `m = s_null`, `d = 1`, `o = v_null` (accumulated in fp32).

2-pass partitioned decode constraint:
- The null token must be included **exactly once** globally.
- The design must not accidentally include the null token once per partition.
- Preferred approach: integrate the null token in the reduction step (treat as a virtual extra “partition” or explicit pre-seed in the reduce kernel).

### R5: Python Plumbing / Feature Exposure
- Remove the runtime guards that prevent fused decode when null-attn is enabled.
- Pass null token tensors from the attention module state into the kernel launches.
- Update fused-path eligibility checks to allow null-attn only when the null-enabled kernels are available.
- Preserve safe fallback behavior: if fused path errors, fall back to streaming decode.

### R6: KV Policy UX (“Intent-First” CLI + One Escape Hatch)
- Provide a unified KV policy string entry point that overrides granular flags.
- The system should detect deprecated/legacy `--kv-cache-*` flags and provide a clear hint: “Use --kv-policy for a unified configuration.”

Decision required (see Open Questions):
- Either expose `--kv-policy` in the minimal CLI (marked advanced), or keep it out of the minimal CLI and support it via config/runner while improving error messaging.

### R7: Validation / Robustness
- Add a feasible long-sequence drift validator that reuses the repository’s existing logit-quality metrics style (rather than hard-coded KL/MSE thresholds that are likely brittle/OOM at 16k).
- Add a CUDA-only parity validation for fused decode with null enabled:
  - Compare fused decode outputs against the existing streaming decode path at small sizes.
  - Cover both 1-pass and 2-pass fused modes.
- Validation must define acceptable numeric tolerances and report failures clearly.

### R8: Run Summaries
- Extend run summaries to include:
  - **KV Policy**: short-string representation of the active policy (when known/forced).
  - **Null Attention**: tri-state indicator: `Inactive`, `Active (Unfused)`, `Active (Fused)`.

## Acceptance Criteria
- Fused decode works with `null_attn=True` for supported quantized cache kinds (q4/q8/q4 as implemented), without correctness regressions.
- When `null_attn=False`, fused decode performance does not regress due to null-token overhead (null logic compiled out).
- 2-pass partitioned fused decode includes the null token exactly once.
- Default training and inference behavior remains unchanged unless the new features are explicitly enabled.
- The known double-`out_proj` bug is fixed.
- Long-seq approximation configuration is explicit and traceable.

## Open Questions (Need User Decision)
1. **CLI exposure**: Should `production/cli.py` expose `--kv-policy` (advanced escape hatch), or should it remain hidden and only be set via `runner_sample.py` / config?
2. **Learned summarizer default**: Keep `mean` as default (recommended for stability), or enable a learned summarizer by default for decoupled training?
3. **Causal gating defaults**: Should the new input-conditioned gating be default-off (recommended) or default-on for new experiments?
4. **Parity tolerances**: What numeric tolerances do you want for fused-vs-streaming parity checks (e.g., max-abs error or relative error thresholds)?

