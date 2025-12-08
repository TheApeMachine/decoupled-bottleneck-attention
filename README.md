# Adaptive Low-Rank Training for Transformers

This repository contains research and experimental code for **Adaptive Low-Rank Training**, a method to dynamically optimize the rank of linear layers in Transformer models during training.

Unlike static compression techniques (like LoRA) or fixed-rank approximations, this approach allows the model to "learn" its own optimal topology. Layers with high information density grow in rank, while redundant layers shrink, automatically allocating parameters where they are most needed.

## 🚀 Latest State: v10 Scaled Spectral Adaptive

The current state-of-the-art implementation in this project is **`v10_transformer_lowrank_scaled.py`**.

### Key Innovations in v10
*   **Stable Rank Estimation**: Instead of heuristic gradient clustering, we use **Spectral Theory**. The "intrinsic rank" is estimated via Stable Rank ($ r_{stable} = \|W\|_F^2 / \sigma_{max}^2 $), where $\sigma_{max}$ is approximated using fast Power Iteration.
*   **Rank Controller**: A sophisticated controller uses Exponential Moving Average (EMA), hysteresis thresholds, and safety factors to prevent rank oscillation and ensure training stability.
*   **SVD-based Resizing**: When a layer's rank changes, weights are projected into the new subspace using Singular Value Decomposition (SVD), preserving learned knowledge.
*   **Layer Heterogeneity**: The model naturally evolves heterogeneous ranks (e.g., Attention layers may compress more than FFN layers), validating the "Intrinsic Dimensionality" hypothesis.

## 📂 Project Structure & Evolution

The codebase documents the evolution of the research idea:

| Version | Description |
|---------|-------------|
| **v10** | **Current Best.** Uses Stable Rank (Spectral) for estimation and SVD for resizing. Implements a robust `RankController`. |
| **v8 - v9** | Introduction of Spectral methods and Bidirectional rank adjustment. |
| **v7** | **Dense Baseline** (`v7_transformer_dense_baseline.py`) and various experiments with EMA and Autograd-based rank adaptation. |
| **v3 - v6** | Early Adaptive Low-Rank implementations. |
| **v1 - v2** | Initial experiments with **Gradient Grouping** and clustering neurons based on similarity. |

## 🛠️ Installation & Requirements

The project relies on standard PyTorch.

```bash
pip install torch torchvision tqdm
```

*Note: `v10` is designed to be hardware-agnostic, automatically selecting CUDA (NVIDIA), MPS (Apple Silicon), or CPU.*

## 🏃 Usage

### 1. Prepare Data
The v10 script expects a tokenized text file (space-separated tokens). A sample `wiki.train.tokens` is expected in the root, or you can point to your own.

### 2. Run the Training
You can run the latest implementation directly or via the `Makefile`.

**Directly:**
```bash
python3 v10_transformer_lowrank_scaled.py \
    --data-file wiki.train.tokens \
    --log-file v10_log.jsonl \
    --epochs 30 \
    --init-rank 64 \
    --min-rank 8 \
    --d-model 512
```

**Using Makefile:**
The `Makefile` contains recipes for reproducing various stages of the research:
```bash
# Run the latest v10 experiment
make train
```

*(Note: You may need to uncomment the specific line in the `Makefile` or run the command manually as shown above.)*

## 📊 Methodology

### The Algorithm
1.  **Forward Pass**: Uses Low-Rank decomposition $ W = U V^T $ where $U \in \mathbb{R}^{d_{out} \times r}, V \in \mathbb{R}^{r \times d_{in}}$. Complexity is $O(r(d_{in} + d_{out}))$.
2.  **Rank Estimation**: Periodically (e.g., every step or epoch), the **Stable Rank** is calculated using a Power Iteration approximation for the top singular value.
3.  **Controller Decision**:
    *   Target rank is smoothed via EMA.
    *   If the target deviates significantly from the current rank (beyond a threshold), a resize is triggered.
4.  **Resizing**:
    *   Reconstruct full $W = U V^T$.
    *   Perform SVD: $W = U' \Sigma V'^T$.
    *   Truncate to new rank $k$.
    *   Re-initialize $U, V$ with truncated components.

### Findings
*   **Phase Transition**: Experiments typically show a "differentiation phase" where ranks adjust rapidly, followed by a stabilization phase where loss drops significantly.
*   **Optimizer Reset**: Changing parameter shapes invalidates optimizer state (momentum). The current implementation re-initializes the optimizer on rank change, which causes temporary loss spikes but allows for architectural flexibility.

## 📄 Research Notes
*   `Adaptive_LowRank_Training_Research.docx.pdf`: Detailed theoretical background.
*   `v10_review_gemini_3_pro_temp_0.md`: AI-assisted review and analysis of the v10 code.

## 📜 License
[MIT](LICENSE)

