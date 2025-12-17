# Run Summary

- Created: `2025-12-17T17:34:09+01:00`
- Out dir: `runs/m4max_decoupled_48_96_seed1337`
- Device: `mps`
- Command: `v29_transformer_decoupled_bottleneck_instrumented.py --mode train --device mps --data fineweb_100m.npy --data-format npy --vocab-size 50257 --steps 6000 --d-model 768 --layers 12 --n-head 12 --d-ff 3072 --embed-dim 512 --optimizer lion --lr 3e-4 --batch-size 8 --grad-accum 2 --train-seq-len 512 --seq-schedule 256@0,512@500,1024@2000 --eval-every 200 --eval-iters 20 --log-every 10 --instrument full --analysis-every 100 --live rich --param-dtype bf16 --amp --amp-dtype bf16 --seed 1337 --out-dir runs/m4max_decoupled_48_96_seed1337 --attn-mode decoupled --attn-dim 144 --sem-dim 48 --geo-dim 96`

## Model Config

```json
{
  "attn_dim": 144,
  "attn_mode": "decoupled",
  "block_size": 256,
  "d_ff": 3072,
  "d_model": 768,
  "dropout": 0.0,
  "embed_dim": 512,
  "geo_dim": 96,
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 12,
  "n_layer": 12,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 48,
  "tie_qk": false,
  "vocab_size": 50257
}
```

## Training Args

```json
{
  "adam_betas": "0.9,0.95",
  "adam_eps": 1e-08,
  "amp": true,
  "amp_dtype": "bf16",
  "analysis_every": 100,
  "analysis_heads": "0",
  "analysis_layers": "0,-1",
  "analysis_local_window": 32,
  "analysis_max_tokens": 256,
  "analysis_save_scores": false,
  "analysis_topk": 8,
  "attn_dim": 144,
  "attn_mode": "decoupled",
  "batch_size": 8,
  "block": 256,
  "ckpt": null,
  "compile": false,
  "compile_mode": "default",
  "d_ff": 3072,
  "d_model": 768,
  "data": "fineweb_100m.npy",
  "data_dtype": "uint16",
  "data_format": "npy",
  "device": "mps",
  "dropout": 0.0,
  "embed_dim": 512,
  "eval_every": 200,
  "eval_iters": 20,
  "eval_seq_len": 0,
  "exp": null,
  "geo_dim": 96,
  "grad_accum": 2,
  "grad_checkpoint": false,
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
  "layers": 12,
  "lion_betas": "0.9,0.99",
  "live": "rich",
  "live_plot": false,
  "live_update_every": 1,
  "log_every": 10,
  "lr": 0.0003,
  "lr_schedule": "constant",
  "matmul_precision": "high",
  "max_new_tokens": 50,
  "min_lr": 0.0,
  "mlp": "swiglu",
  "mode": "train",
  "n_head": 12,
  "no_learned_temp": false,
  "no_null_attn": false,
  "no_rope": false,
  "no_tie_qk": false,
  "null_attn": false,
  "opt_foreach": false,
  "opt_fused": false,
  "optimizer": "lion",
  "out_dir": "runs/m4max_decoupled_48_96_seed1337",
  "param_dtype": "bf16",
  "print_config": false,
  "prompt_tokens": "0",
  "rope": false,
  "rope_base": 10000.0,
  "run_root": "runs",
  "run_tag": null,
  "save_every": 0,
  "seed": 1337,
  "self_opt": "none",
  "self_opt_block_n": "128",
  "self_opt_cache": null,
  "self_opt_calib_decode": 32,
  "self_opt_calib_prefill": 128,
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
  "self_opt_quality_delta_nll_tol": 0.02,
  "self_opt_quality_kl": false,
  "self_opt_quality_kl_tol": null,
  "self_opt_quality_ppl_ratio_tol": 1.02,
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
  "sem_dim": 48,
  "seq_schedule": "256@0,512@500,1024@2000",
  "size": null,
  "steps": 6000,
  "sync_timing": false,
  "tb": false,
  "temperature": 1.0,
  "tie_qk": false,
  "tokenizer": "word",
  "top_k": null,
  "train_seq_len": 512,
  "val_frac": 0.1,
  "vocab_size": 50257,
  "warmup_steps": 0,
  "weight_decay": 0.1
}
```

## Results

- Last step: `6000`
- Best val loss: `5.878566` (ppl `357.30`)
- Files: `train.jsonl`, `analysis.h5` (if enabled), `analysis.png`, `best.pt`, `last.pt`

## KV Cache Memory (batch=1)

- Baseline fp16 (standard attn) @ ctx=256: `9.00MB`
- This run policy @ ctx=256: `1.69MB`
- Compression vs fp16 baseline: `5.33×`
- This run policy @ 128k: `843.8MB`

