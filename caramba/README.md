# caramba

caramba is a **manifest-driven research platform** for building, running, and
iterating on transformer architectures as a system of **small, composable, and
explicit building blocks**.

The core idea is simple:

- **Architectures are graphs** (topologies composed of layers).
- **Graphs are described declaratively** in a manifest (`.yml`/`.yaml`/`.json`).
- **Running an experiment should require nothing more** than supplying a
  manifest.

caramba is the “platform” implementation in this repository: it is not focused
on one model, but on being a clean substrate for rapidly testing new attention
mechanisms, layer wiring, weight transforms, and training loops.

---

## What you get

- **A strict manifest pipeline**: parse → lower → validate → build → run.
- **Explicit module boundaries**:
  - `caramba/config/`: typed config models + preset manifests
  - `caramba/topology/`: graph/topology nodes (stacked/residual/nested/...)
  - `caramba/layer/`: thin torch modules (single concept per file)
  - `caramba/operation/`: pure “math/ops” implementation (RoPE, attention math)
  - `caramba/weight/`: weight modules + guards
  - `caramba/load/`: checkpoint readers + Llama upcycle logic
  - `caramba/trainer/`: orchestration (upcycle, blockwise distillation, etc.)

---

## Install

caramba uses standard Python dependencies (Torch, Pydantic, YAML, HF hub,
safetensors). From repo root:

```bash
python3.12 -m pip install -r requirements.txt
```

Notes:

- If you run on Apple Silicon, **MPS** is supported by default Torch builds.
- If you run on a CUDA machine, install a CUDA-enabled torch separately (avoid
  overwriting it with a plain `pip install torch`).

---

## CLI overview

caramba exposes a minimal CLI through the package entrypoint:

```bash
python3.12 -m caramba --manifest <path/to/manifest.yml>
```

There’s also a compile-only path (parse → lower → validate) with an optional
graph plan print:

```bash
python3.12 -m caramba compile <path/to/manifest.yml> --print-plan
```

---

## Manifests (the contract)

A manifest defines:

- **Model**: embedder + topology (layer graph)
- **Groups**: collections of runs (e.g., upcycle → finetune)
- **Runs**: mode/seed/steps + training config (device/dtype/lr/data/ckpts)

Presets live under:

- `caramba/config/presets/`

If you want to understand how manifests lower into an executable plan, the
compiler lives in:

- `caramba/compiler/`

---

## Upcycling: fitting DBA onto a pretrained Llama

caramba can “upcycle” a pretrained Llama checkpoint into a student model whose
attention blocks have been replaced with **Decoupled Bottleneck Attention
(DBA)**, then train the student using a combination of **blockwise distillation**
and **global fine-tuning**.

This workflow is inspired by *Attention Surgery*:

- Mohsen Ghafoorian, Denis Korzhenkov, Amirhossein Habibian.
  *Attention Surgery: An Efficient Recipe to Linearize Your Video Diffusion
  Transformer.* (2025) \[[arXiv:2509.24899](https://arxiv.org/abs/2509.24899)\]

### 1) Pick the preset

The preset manifest for Llama 3.2 1B → DBA upcycle is:

- `caramba/config/presets/llama32_1b_dba.yml`

### 2) Ensure dataset is present

The preset expects a dataset path (token array) at repo root:

- `fineweb_100m.npy`

If you don’t have it, generate it using the repo helper:

```bash
python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.npy
```

### 3) Ensure HF access is configured

The preset uses an HF URI for the teacher checkpoint:

- `hf://meta-llama/Llama-3.2-1B`

If the repo is gated for your account, authenticate via the Hugging Face CLI
before running:

```bash
huggingface-cli login
```

### 4) Run the upcycle on MPS

Run caramba with the preset manifest:

```bash
python3.12 -m caramba --manifest caramba/config/presets/llama32_1b_dba.yml
```

The preset is configured for:

- **device**: `mps`
- **dtype**: `float32`

If you want a different device or dtype, edit the manifest under:

- `groups[].runs[].train.device`
- `groups[].runs[].train.dtype`

### What happens when you run it

At a high level:

- caramba builds a **teacher** and **student** model from the manifest.
- it loads the teacher checkpoint and applies weights to both models.
- it runs **blockwise distillation** to transplant DBA attention block-by-block.
- it finishes with a **global** phase to fine-tune end-to-end.

If you see long silent periods, it’s typically compute time. caramba prints
coarse stage markers during initialization and blockwise entry to make the
current phase explicit.

---

## Where to look next

- **Architecture diagrams**:
  - `caramba/high-level.png`
  - `caramba/architecture.png`
  - `caramba/llama-architecture.png`
- **Preset manifests**:
  - `caramba/config/presets/`
- **Upcycle trainer**:
  - `caramba/trainer/upcycle.py`
- **Llama checkpoint loader**:
  - `caramba/load/llama_upcycle.py`


