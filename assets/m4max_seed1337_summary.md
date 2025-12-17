# m4max_seed1337 — Visualization Summary

| Run | attn_mode | d_attn | best_val | best_ppl | KV@ctx (MB) | KV@128k (GB) |
|-----|----------:|-------:|---------:|---------:|------------:|------------:|
| baseline | standard | 512 | 6.2961 | 542.5 | 9.00 | 4.39 |
| bottleneck | bottleneck | 144 | 6.5255 | 682.3 | 1.69 | 0.82 |
| gqa | gqa | 768 | 6.2147 | 500.1 | 1.50 | 0.73 |
| decoupled | decoupled | 144 | 5.8786 | 357.3 | 1.69 | 0.82 |

- KV@128k ratio (baseline/bottleneck): **5.33×**
- Best-val delta (bottleneck - baseline): **+0.2294**
- KV@128k ratio (baseline/gqa): **6.00×**
- Best-val delta (gqa - baseline): **-0.0814**
- KV@128k ratio (baseline/decoupled): **5.33×**
- Best-val delta (decoupled - baseline): **-0.4176**
