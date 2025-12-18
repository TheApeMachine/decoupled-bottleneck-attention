### Goal
Make it **one-copy/paste** to kick off a ~**1B parameter** training run (baseline/bottleneck/decoupled/GQA) on a **single CUDA GPU** (RunPod/Lambda/Vast/Colab), using the v29 training script.

This repo already supports:
- **Mixed precision** (`--amp --amp-dtype bf16`, `--param-dtype bf16`)
- **Gradient accumulation** (`--grad-accum`)
- **Gradient checkpointing** (`--grad-checkpoint`)
- **Memory-safe dataset streaming** via `.npy` memmap (`--data-format npy`)

### Recommended hardware
- **Best**: A100/H100 **80GB** (comfortable seq=2048+, faster iteration)
- **Good**: A100 **40GB** / L40S **48GB** (solid single-GPU 1B runs)
- **Possible (slow / tight)**: L4 / 3090 / 4090 **24GB** (reduce seq-len, rely on grad checkpointing + higher grad-accum)
- **Usually not worth it**: T4 **16GB** (1B is very constrained; better to run a ~350M config instead)

### Do you *need* an A100?
No. **A100 is a speed/comfort choice, not a hard requirement** for this repo’s 1B-ish config, because:
- We support **bf16 params + autocast** (cuts model memory ~2x)
- We support **gradient checkpointing** (cuts activation memory)
- We can use **Lion** (1 momentum state, low optimizer memory vs AdamW)

What you *do* need is enough VRAM for activations at your chosen sequence length.

### VRAM tiers: realistic settings
Below are “it should run” settings for the included `run_1b_launch.py` launcher. You can always override further.

- **80GB (A100/H100 80G)**:

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --train-seq-len 2048 --grad-accum 8 --run
```

- **40–48GB (A100 40G / L40S 48G)**:

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --train-seq-len 1024 --seq-schedule "512@0,1024@2000,2048@12000" --grad-accum 16 --run
```

- **24GB (L4 / 4090 / 3090)**:

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --train-seq-len 512 --seq-schedule "512@0,1024@12000" --grad-accum 32 --instrument off --analysis-every 0 --run
```

- **16GB (T4)**: not recommended for 1B. If you must try, start with:

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --train-seq-len 256 --seq-schedule "256@0,512@12000" --grad-accum 64 --instrument off --analysis-every 0 --run
```

If that OOMs, you’ll need to reduce model size (e.g. use the repo’s `--size medium` style configs instead of 1B).

### How long will it take on an 80GB GPU?
Two facts to anchor on:

- **How many tokens you train on** is the main driver of “how well trained” the base model is.
- **Wall time ≈ (trained tokens) / (tokens/sec)**. Your run logs `tok_s` in `runs/.../train.jsonl`, so you can measure this after ~50–200 steps and extrapolate.

Rule of thumb from scaling laws (Chinchilla-style): for a **1B parameter** dense transformer, a “really well trained” base model is often on the order of **~10–20B tokens**.

On a single **A100 80GB**, a realistic training throughput for a 1B-ish model in bf16 is often in the **~10k–25k tokens/sec** range (depends heavily on seq-len, kernels, and how optimized your stack is).

That translates to:
- **1B tokens**: ~11–28 hours
- **10B tokens**: ~4.6–11.6 days
- **20B tokens**: ~9–23 days

The launcher `run_1b_launch.py` now prints `planned_tokens≈...B` for your chosen `--steps/--seq-schedule` so you can sanity-check whether you’re in the “hundreds of millions” vs “tens of billions” regime before you start paying for GPU time.

### Data: get a 1B-token `.npy`
You have three options:
- **Use an existing file**: put `fineweb_1b.npy` + `fineweb_1b.npy.meta` on the machine (or Drive) and point `--data` at it.
- **Generate it on the machine** (slow, but simplest):

```bash
python3.12 prepare_fineweb.py --tokens 1B --output fineweb_1b.npy
```

- **Generate a smaller file first** (recommended sanity check):

```bash
python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy
```

### Install deps (avoid breaking CUDA torch)
On most rented GPU images and Colab, **CUDA-enabled PyTorch is already installed**. Don’t `pip install torch` unless you mean to.

Use the torch-free requirements:

```bash
python3.12 -m pip install -r requirements_runtime.txt
```

### One-command launcher (recommended)
Use the repo helper script `run_1b_launch.py` (added in this repo) to avoid a 40-flag command line.

Dry-run (prints the command):

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy
```

Run it:

```bash
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --run
```

You can also specify a **token budget** and let the launcher solve for steps:

```bash
# Train for ~10B tokens (Chinchilla-ish “decent” for 1B)
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --target-tokens 10B --run

# Train for ~20B tokens (Chinchilla-ish “well trained” for 1B)
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --target-tokens 20B --run
```

### Google Colab quickstart
In a Colab notebook:

```bash
!git clone https://github.com/theapemachine/experiments
%cd experiments
!python3 -m pip install -r requirements_runtime.txt

# Option A: generate data (slow)
#!python3 prepare_fineweb.py --tokens 1B --output fineweb_1b.npy

# Option B: point to a Drive-mounted file:
# from google.colab import drive; drive.mount('/content/drive')
# then set --data "/content/drive/MyDrive/fineweb_1b.npy"

!python3 run_1b_launch.py --device cuda --data fineweb_1b.npy --run
```

### Outputs
Runs write into `runs/` and include:
- `train.jsonl` (metrics)
- `summary.md`
- `best.pt`, `last.pt`

If you want to persist results on Colab, set `--run-root` to a Drive path.


