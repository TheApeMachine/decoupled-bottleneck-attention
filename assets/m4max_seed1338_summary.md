# m4max_seed1338 — Visualization Summary

| Run | attn_mode | d_attn | best_val | best_ppl | KV@ctx (MB) | KV@128k (GB) |
|-----|----------:|-------:|---------:|---------:|------------:|------------:|
| baseline | standard | 512 | 6.6378 | 763.4 | 9.00 | 4.39 |
| bottleneck | bottleneck | 144 | 6.3913 | 596.6 | 1.69 | 0.82 |
| gqa | gqa | 768 | 6.3024 | 545.9 | 1.50 | 0.73 |
| decoupled | decoupled | 144 | 5.8242 | 338.4 | 1.69 | 0.82 |

- KV@128k ratio (baseline/bottleneck): **5.33×**
- Best-val delta (bottleneck - baseline): **-0.2465**
- KV@128k ratio (baseline/gqa): **6.00×**
- Best-val delta (gqa - baseline): **-0.3354**
- KV@128k ratio (baseline/decoupled): **5.33×**
- Best-val delta (decoupled - baseline): **-0.8136**
