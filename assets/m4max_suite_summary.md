# v29 Suite Summary

- tag: `m4max`
- seeds: `1337,1338`

## Best val loss / ppl (per seed)

| variant | seed | best_val_loss | best_ppl | kv@128k | run_dir |
|---|---:|---:|---:|---:|---|
| baseline | 1337 | 6.296133 | 542.47 | 4.39GB | `runs/m4max_baseline_seed1337` |
| baseline | 1338 | 6.637763 | 763.39 | 4.39GB | `runs/m4max_baseline_seed1338` |
| gqa_kv2 | 1337 | 6.214722 | 500.06 | 750.0MB | `runs/m4max_gqa_kv2_seed1337` |
| gqa_kv2 | 1338 | 6.302364 | 545.86 | 750.0MB | `runs/m4max_gqa_kv2_seed1338` |
| bottleneck_144 | 1337 | 6.525506 | 682.32 | 843.8MB | `runs/m4max_bottleneck_144_seed1337` |
| bottleneck_144 | 1338 | 6.391262 | 596.61 | 843.8MB | `runs/m4max_bottleneck_144_seed1338` |
| decoupled_48_96 | 1337 | 5.878566 | 357.30 | 843.8MB | `runs/m4max_decoupled_48_96_seed1337` |
| decoupled_48_96 | 1338 | 5.824183 | 338.38 | 843.8MB | `runs/m4max_decoupled_48_96_seed1338` |

## Aggregate across seeds (mean ± std)

| variant | best_val_loss | best_ppl |
|---|---:|---:|
| baseline | 6.467 ± 0.242 | 652.9 ± 156.2 |
| gqa_kv2 | 6.259 ± 0.062 | 523.0 ± 32.4 |
| bottleneck_144 | 6.458 ± 0.095 | 639.5 ± 60.6 |
| decoupled_48_96 | 5.851 ± 0.038 | 347.8 ± 13.4 |

