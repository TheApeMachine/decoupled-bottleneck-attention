# Run Summary

- Created: `2025-12-16T16:02:41+01:00`
- Out dir: `runs/tiny_bottleneck`
- Device: `mps`
- Command: `v27_transformer_decoupled_bottleneck_instrumented.py --mode train --data fineweb_100m.tokens --data-format text --size tiny --exp paper_bottleneck --live-plot`
- Size preset: `tiny`
- Experiment: `paper_bottleneck`

## Model Config

```json
{
  "attn_dim": 96,
  "attn_mode": "bottleneck",
  "block_size": 1024,
  "d_ff": 2048,
  "d_model": 512,
  "dropout": 0.0,
  "embed_dim": 512,
  "geo_dim": 64,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 8,
  "n_layer": 6,
  "null_attn": true,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 32,
  "tie_qk": false,
  "vocab_size": 50257
}
```

## Training Args

```json
{
  "amp": false,
  "amp_dtype": "bf16",
  "analysis_every": 100,
  "analysis_heads": "0",
  "analysis_layers": "0,-1",
  "analysis_local_window": 32,
  "analysis_max_tokens": 256,
  "analysis_save_scores": false,
  "analysis_topk": 8,
  "attn_dim": 96,
  "attn_mode": "bottleneck",
  "batch_size": 16,
  "block": 1024,
  "ckpt": null,
  "compile": false,
  "compile_mode": "default",
  "d_ff": 2048,
  "d_model": 512,
  "data": "fineweb_100m.tokens",
  "data_dtype": "uint16",
  "data_format": "text",
  "device": null,
  "dropout": 0.0,
  "embed_dim": 512,
  "eval_every": 200,
  "eval_iters": 20,
  "exp": "paper_bottleneck",
  "geo_dim": 64,
  "grad_clip": 1.0,
  "instrument": "full",
  "kv_cache": "fp16",
  "kv_cache_k": null,
  "kv_cache_k_geo": null,
  "kv_cache_k_sem": null,
  "kv_cache_v": null,
  "kv_decode_block": 1024,
  "kv_fused": "auto",
  "kv_head": null,
  "kv_qblock": 32,
  "kv_qblock_k": null,
  "kv_qblock_k_geo": null,
  "kv_qblock_k_sem": null,
  "kv_qblock_v": null,
  "kv_residual": 128,
  "layers": 6,
  "live_plot": true,
  "log_every": 100,
  "lr": 0.0003,
  "max_new_tokens": 50,
  "mlp": "swiglu",
  "mode": "train",
  "n_head": 8,
  "no_learned_temp": false,
  "no_null_attn": false,
  "no_rope": false,
  "no_tie_qk": false,
  "null_attn": true,
  "out_dir": "runs/tiny_bottleneck",
  "print_config": false,
  "prompt_tokens": "0",
  "rope": false,
  "rope_base": 10000.0,
  "run_root": "runs",
  "run_tag": null,
  "seed": 1337,
  "self_opt": "none",
  "self_opt_block_n": "128",
  "self_opt_cache": null,
  "self_opt_calib_decode": 8,
  "self_opt_calib_prefill": 64,
  "self_opt_calib_tokens": null,
  "self_opt_decode_blocks": "256,512,1024,2048",
  "self_opt_hysteresis": 0.03,
  "self_opt_interval": 256,
  "self_opt_iters": 3,
  "self_opt_k_geo_kinds": "q8_0,q4_0,fp16",
  "self_opt_k_sem_kinds": "q4_0,nf4,q8_0,fp16",
  "self_opt_mem_budget_mb": null,
  "self_opt_mem_overhead_frac": 0.1,
  "self_opt_policy_hysteresis": 0.02,
  "self_opt_policy_iters": 3,
  "self_opt_policy_prefix_len": null,
  "self_opt_policy_quality": false,
  "self_opt_policy_warmup": 1,
  "self_opt_prefer_low_mem_within": 0.02,
  "self_opt_qblocks": "16,32,64",
  "self_opt_quality_tol": 0.5,
  "self_opt_residuals": "0,32,64,128",
  "self_opt_scope": "all",
  "self_opt_stages": "2,3",
  "self_opt_v_kinds": "q4_0,q8_0,fp16",
  "self_opt_verbose": false,
  "self_opt_verify": false,
  "self_opt_verify_tol": 0.005,
  "self_opt_warmup": 1,
  "self_opt_warps": "4,8",
  "sem_dim": 32,
  "size": "tiny",
  "steps": 6000,
  "tb": false,
  "temperature": 1.0,
  "tie_qk": false,
  "tokenizer": "word",
  "top_k": null,
  "val_frac": 0.1,
  "vocab_size": null,
  "weight_decay": 0.1
}
```

## Results

- Last step: `6000`
- Best val loss: `4.334908` (ppl `76.32`)
- Files: `train.jsonl`, `analysis.h5` (if enabled), `analysis.png`, `best.pt`, `last.pt`

## KV Cache Memory (batch=1)

- Baseline fp16 (standard attn) @ ctx=1024: `12.00MB`
- This run policy @ ctx=1024: `2.25MB`
- Compression vs fp16 baseline: `5.33×`
- This run policy @ 128k: `281.2MB`

