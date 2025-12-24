# Decoupled Bottleneck Attention

**Scaling Efficient Transformers via Low-Rank Semantic Routing**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![Paper](https://img.shields.io/badge/PDF-paper.pdf-blue)](paper.pdf)

---

> This repository now also contains an evolution of the original production implementation, which is a full, more modular rewrite, called [caramba](caramba/README.md). It is being developed as a set of composable atomic building blocks to rapidly experiment with new architectures and optimizations, and contains some advanced features.
> One of the most interesting features the [caramba](caramba/README.md) implementation provides is the ability to "upcycle" a Llama model, retro-fitting Decoupled Bottleneck Attention via a method inspired by *Attention Surgery* (Ghafoorian et al., 2025) \[[arXiv:2509.24899](https://arxiv.org/abs/2509.24899)\].

## TL;DR

This repo contains the **production implementation** and **paper reproduction harness** for *Decoupled Bottleneck Attention*.

- **Single source of truth**: `production/` (invoked via `main.py`)
- **Single dataset**: **FineWeb-Edu GPT-2 tokens** (`fineweb_100m.npy`, `fineweb_20b.npy`)
- **Single reproduction entrypoint**: `paper_manifest.json` + `run_paper_manifest.py`

Paper numbers and plots are generated from run artifacts into `assets/paper_results.json` by `generate_paper_figures.py`.

**Key Insight:** *Attention is a router, not a processor.* Semantic routing operates in ~32 dimensions; positional geometry needs ~64. By decoupling them, we slash memory while preserving quality.

---

## Key Results (how we measure)

- **Training/eval metrics**: written to `runs/<id>/train.jsonl` by the production runner.
- **KV-cache @128k memory**: written to `runs/<id>/mem128k.json` by `production/bench_end_to_end_memory.py`.
  - For decoupled runs we also report the paper decomposition:
    - **architecture-only**: standard FP16 → decoupled FP16
    - **quant-only**: decoupled FP16 → hetero Q4/Q8/Q4
    - **end-to-end**: standard FP16 → hetero Q4/Q8/Q4
- **Paper artifacts**: `generate_paper_figures.py` produces:
  - `assets/paper_results.json`
  - `assets/fig_convergence.png`
  - `assets/fig_pareto_memory_vs_loss.png`
  - `assets/table_main.tex`, `assets/table_scale.tex`

---

## Quick Start (reviewer reproduction)

### Prerequisites

You need Python 3.10+ and PyTorch for either CUDA or MPS.

```bash
python -m pip install -r requirements.txt
```

Notes:
- On CUDA machines, install a CUDA-enabled torch separately (do not `pip install torch` over it).
- `matplotlib` is optional unless you want to generate plots.

### Dataset files (FineWeb-Edu tokens)

The harness expects tokenized arrays:
- `fineweb_100m.npy` (local suite)
- `fineweb_20b.npy` (A100 scale suite)

If you don’t have them, generate them with:

```bash
python prepare_fineweb.py --tokens 100M --output fineweb_100m.npy
```

Note: the A100 suite expects `fineweb_20b.npy`. If you already have a prebuilt 20B-token shard on your A100 instance,
copy it into the repo root under that exact filename.

### One-command reproduction (manifest-driven)

All paper runs are defined in `paper_manifest.json`. The harness:
- validates resolved configs (no flag ambiguity),
- writes per-run provenance (`command.txt`, `resolved_config.json`, `resolved_run.json`),
- runs training,
- optionally runs `mem128k.json` benchmarking after training.

#### 1) Validate configs (no training)

```bash
python run_paper_manifest.py --group mac_fw100m --dry-run
python run_paper_manifest.py --group a100_fw1b_1bscale --dry-run
```

#### 2) Run the Mac suite (FineWeb 100M)

```bash
python run_paper_manifest.py --group mac_fw100m --post-mem128k
```

#### 3) Run the A100 suite (FineWeb 20B tokens)

```bash
python run_paper_manifest.py --group a100_fw1b_1bscale --post-mem128k
```

The A100 runs are **resumable**. Re-running the same command will resume from `runs/<id>/last.pt` if present.

---

## Repository Structure

```
experiments/
├── production/                                 # Canonical implementation (single source of truth)
├── main.py                                     # Canonical CLI entrypoint (calls production/cli.py)
├── paper_manifest.json                         # Canonical paper run manifest
├── run_paper_manifest.py                       # Canonical paper harness runner
├── generate_paper_figures.py                   # Generates assets/ paper artifacts from run dirs
├── paper.tex                                   # Paper source (inputs assets/table_*.tex, assets/*.png)
├── references.bib                              # Bibliography
├── Makefile                                    # Misc helpers (not used for paper reproduction)
├── experiments/                                # Historical experiments (not paper-canonical)
├── runs/                                       # Training logs (and checkpoints once run locally)
├── assets/                                     # Generated figures
│   ├── paper_results.json
│   ├── fig_convergence.png
│   ├── fig_pareto_memory_vs_loss.png
│   ├── table_main.tex
│   └── table_scale.tex
└── docs/
    └── RESEARCH_MASTER_PLAN.md
```

---

## Paper experiment contract (what reviewers should check)

For any run id in `paper_manifest.json`, reviewers should be able to verify:

- **Provenance**
  - `runs/<id>/command.txt` matches what was executed
  - `runs/<id>/resolved_config.json` contains the fully resolved config used
  - `runs/<id>/train.jsonl` contains `meta`/`resume_meta`, `train`, `eval`, `done` events
- **Memory measurement**
  - `runs/<id>/mem128k.json` exists (when using `--post-mem128k`)
  - decoupled runs include `decomposition.estimate_bytes` and `decomposition.measured`

---

## Generate paper figures / tables

After running experiments:

```bash
python generate_paper_figures.py
```

This writes:
- `assets/paper_results.json`
- `assets/fig_convergence.png`
- `assets/fig_pareto_memory_vs_loss.png`
- `assets/table_main.tex`, `assets/table_scale.tex`

---

## Minimal CLI (intent-first)

This project now defaults to a **minimal, self-optimizing CLI**: you specify *intent* (train/sample, model size, data, instrumentation) and the system self-tunes performance/kv-cache policies based on your hardware and workload.

### Train (example)

```bash
python3 main.py --mode train --size medium --exp paper_decoupled --data fineweb_100m.npy
```

### Sample (example)

```bash
python3 main.py --mode sample --ckpt runs/<run_id>/best.pt --prompt-tokens "0 1 2 3" --max-new-tokens 64
```

### Debug/repro

Optimization behavior is fully self-driven; there are no environment-variable toggles for disabling core optimization paths.

## Model Configuration

Model/training recipes are defined via presets and manifests:

- Presets: `production/config.py`
- Paper harness: `run_paper_manifest.py` + `paper_manifest.json`

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

### Related work

If you reference the *Attention Surgery* method discussed above:

```bibtex
@article{ghafoorian2025attentionsurgery,
  title={Attention Surgery: An Efficient Recipe to Linearize Your Video Diffusion Transformer},
  author={Ghafoorian, Mohsen and Korzhenkov, Denis and Habibian, Amirhossein},
  journal={arXiv preprint arXiv:2509.24899},
  year={2025},
  doi={10.48550/arXiv.2509.24899}
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
