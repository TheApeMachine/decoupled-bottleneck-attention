# Run Summary

- Created: `2025-12-19T00:45:24+01:00`
- Out dir: `runs/1b_train_decoupled_fast`
- Device: `mps`
- Command: `/Users/theapemachine/go/src/github.com/theapemachine/experiments/production/cli.py --mode train --size 1b --exp train_decoupled_fast --data fineweb_1b.npy --data-format npy --vocab-size 50257 --block 4096 --train-seq-len 2048 --seq-schedule 512@0,1024@200,2048@600,3072@1200,4096@2000 --steps 20000 --log-every 10 --eval-every 200 --eval-iters 5 --eval-seq-len 512 --analysis-every 0 --instrument basic --batch-size 8 --grad-accum 8 --wandb --wandb-entity p4n0p71c0n --wandb-project 1b_decoupled --wandb-name 1b_decoupled --wandb-mode online --no-learned-temp --nan-policy error`

## Model Config

```json
{
  "attn_dim": 768,
  "attn_mode": "decoupled",
  "block_size": 4096,
  "d_ff": 8192,
  "d_model": 2048,
  "decoupled_gate": true,
  "dropout": 0.0,
  "embed_dim": 2048,
  "geo_dim": 512,
  "kv_head": null,
  "learned_temp": false,
  "mlp": "swiglu",
  "n_head": 16,
  "n_layer": 18,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 256,
  "tie_qk": true,
  "vocab_size": 50257
}
```

