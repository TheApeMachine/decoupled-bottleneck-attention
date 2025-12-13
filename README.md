Here is a professional, high-impact `README.md` ready for your repository. It highlights the key findings immediately and makes reproduction simple.

You will need to place two images in your repo for this to look perfect:
1.  `assets/early_convergence.png` (Screenshot of Figure 1 from your PDF)
2.  `assets/pareto_curve.png` (Screenshot of Figure 2 from your PDF)

***

```markdown
# Bottleneck Attention: Hard-Wiring Low-Dimensional Routing

**[Read the Technical Report (PDF)](bottleneck_attention_tech_report.pdf)**

### TL;DR
Transformer attention matrices ($W_Q, W_K$) are massively over-parameterized. We demonstrate that you can **reduce the attention dimension from 512 to 32 (16$\times$ compression)** with only a ~4% increase in perplexity.

This architectural change reduces the **KV-Cache memory footprint by 93.75%**, enabling significantly larger batch sizes and longer contexts during inference.

---

## 📊 Key Results

We trained a 6-layer GPT-style model on WikiText-2 using various attention dimensions ($d_{attn}$). The residual stream ($d_{model}$) remained constant at 512.

| Model Config | $d_{attn}$ | Params | Compression | Best Val Loss | vs Baseline | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 512 | 36.06M | 1.0x | 5.3688 | -- | ~540 tok/s |
| **Bottleneck** | 128 | 31.34M | 4.0x | 5.4686 | +1.8% | ~600 tok/s |
| **Extreme + Null** | **32** | **30.16M** | **16.0x** | **5.6070** | **+4.4%** | **~626 tok/s** |

### The "Early Convergence" Anomaly
Surprisingly, restricting the attention dimension actually **accelerates early training**. In the first 500 steps, the Bottleneck models (Rank 128 and 32) achieve lower perplexity than the full-rank Baseline, suggesting that the standard 512-dimensional initialization introduces significant optimization noise.

![Early Convergence](assets/early_convergence.png)
*(Figure 1: Validation perplexity vs training step. Note the bottleneck models diving under the baseline in the inset.)*

---

## 🧠 The Hypothesis: "Wide Stream, Narrow Router"

Standard Transformers assume $d_{attn} = d_{model}$.
Our findings suggest this is inefficient. Attention acts as a **routing primitive** (deciding *where* to move information), which is an intrinsically low-rank operation (Rank $\approx$ 11-32). Feature processing, handled by the MLPs, requires high rank.

**Bottleneck Attention** hard-wires this by decoupling the dimensions:
1.  Project $d_{model} \to d_{attn}$ (where $d_{attn} \ll d_{model}$).
2.  Compute Attention Scores and Aggregation in $d_{attn}$.
3.  Project $d_{attn} \to d_{model}$.

**The Null Token:**
To prevent the "Softmax Bottleneck" in low dimensions, we add a learnable `null` key/value vector. This allows the model to explicitly assign probability mass to "attend nowhere," preventing noise from contaminating the residual stream when no relevant context exists.

---

## 🚀 Reproduction

We provide a self-contained training script that requires only PyTorch. The dataset (WikiText-2) is downloaded and tokenized automatically.

### Prerequisites
```bash
pip install torch
```

### Run Experiments
To reproduce the Baseline, Rank 128, and Rank 32 experiments sequentially:

```bash
make bottleneck_attention
```

*Note: This runs the script `v19_transformer_attn_bottleneck.py`. Logs and checkpoints will be saved to `runs/`.*

---

## 📦 Inference Implications

The primary benefit of this architecture is **Inference Memory**.
Standard KV Cache size is $2 \cdot L \cdot T \cdot d_{model}$.
Bottleneck KV Cache size is $2 \cdot L \cdot T \cdot d_{attn}$.

| Context Length | Standard Cache ($d=512$) | Bottleneck Cache ($d=32$) | Savings |
| :--- | :--- | :--- | :--- |
| **4k** | ~10 MB | ~0.6 MB | **93%** |
| **128k** | ~320 MB | ~20 MB | **93%** |

This effectively allows for **16$\times$ longer context** or **16$\times$ larger batch sizes** within the same memory budget for the KV cache.

---

## 📜 Citation

If you find this useful, please cite the technical report:

```bibtex
@techreport{vandommelen2025bottleneck,
  title={Bottleneck Attention: Hard-Wiring Low-Dimensional Routing in GPT-Style Transformers},
  author={van Dommelen, Daniel Owen},
  year={2025},
  month={December},
  institution={Independent Research},
  url={https://github.com/TheApeMachine/bottleneck-attention}
}
```
```