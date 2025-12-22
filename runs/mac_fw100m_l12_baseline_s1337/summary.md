# Run Summary

- Created: `2025-12-22T15:51:14+01:00`
- Out dir: `runs/mac_fw100m_l12_baseline_s1337`
- Device: `mps`
- Command: `main.py --mode train --exp paper_baseline --data fineweb_100m.npy --out-dir runs/mac_fw100m_l12_baseline_s1337 --seed 1337 --steps 6000`

## Model Config

```json
{
  "attn_dim": 104,
  "attn_mode": "standard",
  "block_size": 512,
  "d_ff": 600,
  "d_model": 128,
  "decoupled_gate": true,
  "device": "mps",
  "dim_multiplier": 0,
  "dropout": 0.0,
  "embed_dim": 32,
  "geo_dim": 0,
  "head_dim": 32,
  "head_policy": "standard",
  "kv_head": null,
  "learned_temp": true,
  "mlp": "swiglu",
  "n_head": 4,
  "n_layer": 12,
  "null_attn": false,
  "rope": true,
  "rope_base": 10000.0,
  "sem_dim": 104,
  "tie_qk": false,
  "vocab_size": 50257
}
```

