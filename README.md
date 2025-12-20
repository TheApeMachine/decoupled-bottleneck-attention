# Decoupled Bottleneck Attention

**Scaling Efficient Transformers via Low-Rank Semantic Routing**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![Paper](https://img.shields.io/badge/PDF-paper.pdf-blue)](paper.pdf)

---

## 🎯 TL;DR

Transformer attention is **5× over-parameterized**. We reduce attention dimension from 512 to 96 and achieve **better perplexity** while enabling **168× KV-cache compression**.

| Model              | Attn Dim | Val Loss | Memory (128k ctx) |
|:-------------------|:--------:|:--------:|:-----------------:|
| Standard Baseline  | 512      | 5.37     | 64 GB             |
| **Bottleneck 96**  | 96       | **5.33** | 1.5 GB            |
| **Decoupled + Q4** | 32+64    | 5.59     | **0.38 GB**       |

**Key Insight:** *Attention is a router, not a processor.* Semantic routing operates in ~32 dimensions; positional geometry needs ~64. By decoupling them, we slash memory while preserving quality.

---

## 📊 Key Results

### The "Bottleneck Beats Baseline" Finding

Surprisingly, constraining attention to 96 dimensions **outperforms** full-rank 512-dim attention:

![Convergence Plot](assets/convergence_plot.png)

### Memory Footprint at 128k Context (Llama-7B Scale)

![Memory Footprint](assets/memory_footprint.png)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+ required
pip install torch numpy matplotlib seaborn tqdm

# Or use the Makefile:
make install_deps
```

### Train a ~1B model on a rented GPU / Colab (CUDA)

If you want the *easiest possible* kick-off for a single-GPU ~1B run, use the helper launcher:

```bash
# Install deps without touching your CUDA-enabled torch install:
python3.12 -m pip install -r requirements_runtime.txt

# (Optional) prepare data
python3.12 prepare_fineweb.py --tokens 1B --output fineweb_1b.npy

# Dry-run (prints the full v29 command):
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy

# Actually run:
python3.12 run_1b_launch.py --device cuda --data fineweb_1b.npy --run
```

More details + Colab notes are in `docs/LAUNCH_1B.md`.

### Setup

```bash
# Prepare WikiText-2 dataset (auto-downloads if needed)
make setup
```

### One-Command Replication

To reproduce **all experiments from the paper**:

```bash
make replicate_paper
```

This automatically:

1. Runs `setup` to prepare WikiText-2
2. Runs WikiText-2 core experiments
3. Runs ablation studies
4. **Downloads and prepares FineWeb-Edu (~448MB)** if not present
5. Runs FineWeb validation experiments
6. Generates all figures

This runs the full experimental suite (~8-12 hours on M1/M2 Mac or CUDA GPU):

1. WikiText-2 baseline and ablations
2. GQA comparison
3. Long-context stress tests
4. FineWeb-Edu validation (if `fineweb_100m.tokens` exists)

---

## 📁 Repository Structure

```
decoupled-bottleneck-attention/
├── v19_transformer_attn_bottleneck.py          # Tech report training script
├── v21_transformer_decoupled_bottleneck.py     # Main training script
├── v21_transformer_decoupled_bottleneck_gqa.py # Main training script
├── v22_decoupled_bottleneck_survive_scale.py   # Production grade/Datacenter version
├── bottleneck_attention_tech_report.pdf        # Tech report on preceeding research/results
├── decoupled_bottleneck_attention.pdf          # PDF paper
├── paper.tex                                   # LaTeX paper source
├── references.bib                              # Bibliography
├── Makefile                                    # Experiment commands
├── plot_memory.py                              # Memory visualization
├── plot_results.py                             # Result graphs
├── vis_heatmap.py                              # Attention heatmaps
├── experiments/                                # Every historical version of the code from beginning to now
├── runs/                                       # Training logs (and checkpoints once run locally)
├── assets/                                     # Generated figures
│   ├── convergence_plot.png
│   ├── memory_footprint.png
│   └── pareto_curve.png
│   └── ...
└── docs/
    └── RESEARCH_MASTER_PLAN.md
```

---

## 🧪 Experiment Guide

### Core Experiments (WikiText-2)

#### 1. Combined Bottleneck 96 (Best Perplexity)

```bash
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_combined_baseline_96 \
    --attn-mode bottleneck \
    --attn-dim 96 \
    --null-attn
```

#### 2. Decoupled Bottleneck (Semantic 32 + Geometric 64)

```bash
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_decoupled_sem32_geo64 \
    --attn-mode decoupled \
    --sem-dim 32 \
    --geo-dim 64 \
    --attn-dim 128 \
    --embed-dim 512 \
    --tie-qk \
    --null-attn
```

#### 3. Standard Baseline (Full Rank 512)

```bash
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_baseline \
    --attn-mode standard \
    --d-model 512
```

### Ablation Studies

#### GQA Comparison (8 Query / 2 KV Heads)

```bash
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_gqa_kv2_parammatch \
    --attn-mode gqa \
    --kv-head 2 \
    --d-model 512 \
    --attn-dim 128 \
    --null-attn \
    --batch-size 64
```

#### Small Model Control (d_model=128)

Proves the "wide residual stream" hypothesis—you can't just shrink everything:

```bash
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_small_d128_standard \
    --attn-mode standard \
    --d-model 128 \
    --layers 6 \
    --n-head 4 \
    --d-ff 512 \
    --embed-dim 128 \
    --null-attn \
    --batch-size 64
```

#### Long Context Stress Tests

```bash
# 1024 context
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_decoupled_block1024 \
    --attn-mode decoupled \
    --sem-dim 32 --geo-dim 64 \
    --block 1024 \
    --batch-size 8 \
    --steps 1200

# 2048 context
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data wiki.train.tokens \
    --out-dir runs/v21_decoupled_block2048 \
    --attn-mode decoupled \
    --sem-dim 32 --geo-dim 64 \
    --block 2048 \
    --batch-size 4 \
    --steps 800
```

### FineWeb-Edu (Large Scale Validation)

```bash
# Step 1: Prepare dataset (downloads ~2GB)
python3 prepare_fineweb.py --out fineweb_100m.tokens

# Step 2: Run baseline
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data fineweb_100m.tokens \
    --out-dir runs/v21_fineweb_baseline \
    --attn-mode standard \
    --d-model 512 \
    --block 1024 \
    --batch-size 16 \
    --steps 6000

# Step 3: Run decoupled
python3 v21_transformer_decoupled_bottleneck_gqa.py \
    --data fineweb_100m.tokens \
    --out-dir runs/v21_fineweb_decoupled \
    --attn-mode decoupled \
    --sem-dim 32 --geo-dim 64 \
    --block 1024 \
    --batch-size 16 \
    --tie-qk --null-attn \
    --steps 6000
```

---

## 📈 Generate Figures

After running experiments:

```bash
# Convergence curves
python3 plot_results.py

# Memory footprint chart
python3 plot_memory.py

# Attention heatmaps (requires checkpoint)
python3 vis_heatmap.py --ckpt runs/v21_combined_baseline_96/best.pt
```

---

## 🔧 Model Configuration

| Argument      | Default    | Description                                  |
|:--------------|:-----------|:---------------------------------------------|
| `--attn-mode` | `standard` | `standard`, `bottleneck`, `decoupled`, `gqa` |
| `--d-model`   | `512`      | Residual stream width                        |
| `--attn-dim`  | `512`      | Attention bottleneck dimension               |
| `--sem-dim`   | `32`       | Semantic path dimension (decoupled mode)     |
| `--geo-dim`   | `64`       | Geometric path dimension (decoupled mode)    |
| `--null-attn` | `False`    | Enable learnable null token                  |
| `--tie-qk`    | `False`    | Tie Q and K projections (semantic path)      |
| `--kv-head`   | `None`     | KV heads for GQA mode                        |
| `--block`     | `256`      | Context length                               |
| `--layers`    | `6`        | Number of transformer layers                 |
| `--n-head`    | `8`        | Number of attention heads                    |

### Decoupled KV cache: semantic vs geometric precision (why k_geo is usually higher precision)

In **decoupled attention**, attention logits are the sum of two paths:

- **Semantic path**: content similarity (no RoPE)
- **Geometric path**: relative positional similarity (**RoPE applied here only**)

Because RoPE encodes a geometric/rotational positional signal, the **geometric K/V state is often more sensitive to quantization error**. In practice this is why default heterogeneous KV-cache policies commonly use **more aggressive quantization on `k_sem`** (e.g. `q4_0`/`nf4`) while keeping **`k_geo` at higher precision** (e.g. `q8_0` or `fp16`) to preserve positional fidelity over long contexts.

The production self-optimizer can still explore counterfactual configurations (including sem/geo swaps), but policy acceptance is guarded by short- and optional long-horizon quality checks vs an fp16-cache baseline.
---

## 📜 Citation

```bibtex
@article{vandommelen2025decoupled,
  title={Decoupled Bottleneck Attention: Scaling Efficient Transformers via Low-Rank Semantic Routing},
  author={van Dommelen, Daniel Owen},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

---

## 🙏 Acknowledgments

This work builds on insights from:
- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) — Low-rank adaptation
- [AdaRankGrad](https://arxiv.org/abs/2410.17881) (Refael et al., 2024) — Gradient rank dynamics
- [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434) — Multi-head latent attention
- [ExLlamaV2](https://github.com/turboderp/exllamav2) — 4-bit KV cache quantization
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Production Q4_0 implementation

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
