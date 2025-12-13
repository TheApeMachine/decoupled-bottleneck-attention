#!/usr/bin/env python3
# v15_transformer_lowrank_adaptive_grad.py

import math
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS backend (MPS).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    print("Using CPU backend.")
    return torch.device("cpu")


device = get_device()

# ============================================================
# SIMPLE JSONL LOGGER
# ============================================================

class JsonLogger:
    def __init__(self, path: str | None):
        if path is None:
            self.f = None
        else:
            p = Path(path)
            # Start fresh each run
            p.unlink(missing_ok=True)
            self.f = p.open("a", encoding="utf-8")

    def log(self, obj: dict):
        if self.f is None:
            return
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()


# ============================================================
# DATASET LOADING (WIKITEXT-2 TOKEN FILE STYLE)
# ============================================================

def load_token_file(path: str, train_frac: float = 0.9, tokenizer_model: str = "gpt2"):
    """
    Loads data from a file. Detects if it's pre-tokenized (integers) or raw text.
    For raw text, uses tiktoken (GPT-2) or falls back to character-level tokenization.
    We split contiguously into train/val by train_frac.
    """
    txt = Path(path).read_text(encoding="utf-8")
    
    # Check if it looks like space-separated integers (pre-tokenized)
    # We look at the first 50 tokens to guess
    sample_tokens = txt.strip().split(maxsplit=50)
    is_pretokenized = False
    if len(sample_tokens) > 0:
        try:
            [int(t) for t in sample_tokens]
            is_pretokenized = True
        except ValueError:
            is_pretokenized = False
    
    if is_pretokenized:
        print("Detected pre-tokenized integer data.")
        # Load all tokens
        ids = [int(t) for t in txt.strip().split()]
        data = torch.tensor(ids, dtype=torch.long)
        vocab_size = int(data.max().item()) + 1
        print(f"Vocab size (from max token id): {vocab_size}")
    else:
        print(f"Detected raw text. Using tiktoken ({tokenizer_model})...")
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_model)
            ids = enc.encode(txt, allowed_special={'<|endoftext|>'})
            data = torch.tensor(ids, dtype=torch.long)
            vocab_size = enc.n_vocab
            print(f"Vocab size (tiktoken): {vocab_size}")
        except ImportError:
            print("\n[ERROR] 'tiktoken' not found. Please install it: pip install tiktoken")
            print("Falling back to simple character-level tokenization.")
            chars = sorted(list(set(txt)))
            stoi = {c: i for i, c in enumerate(chars)}
            ids = [stoi[c] for c in txt]
            data = torch.tensor(ids, dtype=torch.long)
            vocab_size = len(chars)
            print(f"Vocab size (char-level): {vocab_size}")
    
    n = int(len(data) * train_frac)
    train = data[:n]
    val = data[n:]
    print(f"Train tokens: {len(train)}, Val tokens: {len(val)}")
    return train.to(device), val.to(device), vocab_size


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    """
    Random contiguous subsequences from flat token stream.
    """
    n = data.size(0)
    assert n > block_size + 1
    ix = torch.randint(0, n - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ============================================================
# SPECTRAL RANK CONTROLLER (GRADIENT-BASED)
# ============================================================

class SpectralRankController:
    """
    Maintains an adaptive rank based on gradient spectrum.

    - Uses gradient SVD singular values s.
    - Energy = sum s_i^2; picks smallest r with cumulative >= energy_target * total.
    - Smooths with EMA and clamps per-update rank change.
    """

    def __init__(
        self,
        init_rank: int,
        min_rank: int,
        max_rank: int,
        energy_target: float = 0.99,
        ema_decay: float = 0.8,
        max_step_down: int = 8,
        max_step_up: int = 4,
    ):
        self.rank = int(init_rank)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.energy_target = energy_target
        self.ema_decay = ema_decay
        self.max_step_down = max_step_down
        self.max_step_up = max_step_up

        self._ema_target = float(init_rank)

    def propose_from_svals(self, svals: torch.Tensor):
        """
        Given 1D singular values (sorted desc), return:
        - spectral_rank (smallest r meeting energy_target)
        - tail_energy (remaining energy fraction)
        """
        if svals.numel() == 0:
            return self.rank, 1.0

        energy = svals.pow(2)
        total = energy.sum()
        if total <= 0:
            return self.rank, 1.0

        cum = torch.cumsum(energy, dim=0)
        target = self.energy_target * total

        idx = (cum >= target).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            r = svals.numel()
        else:
            r = int(idx[0].item() + 1)

        r = max(self.min_rank, min(r, self.max_rank))
        tail = float((total - cum[r - 1]) / total)
        return r, tail

    def update(self, spectral_rank: int):
        """
        Combine spectral suggestion with EMA & step caps.
        """
        spectral_rank = max(self.min_rank, min(int(spectral_rank), self.max_rank))
        # EMA target
        self._ema_target = (
            self.ema_decay * self._ema_target + (1.0 - self.ema_decay) * spectral_rank
        )
        target = int(round(self._ema_target))

        # Clamp step size
        delta = target - self.rank
        if delta < 0:
            delta = max(delta, -self.max_step_down)
        else:
            delta = min(delta, self.max_step_up)

        new_rank = int(self.rank + delta)
        new_rank = max(self.min_rank, min(new_rank, self.max_rank))
        old_rank = self.rank
        self.rank = new_rank
        return old_rank, new_rank


# ============================================================
# VIRTUAL LOW-RANK LINEAR (FACTOR BUFFERS + GRADIENT SPECTRUM)
# ============================================================

class VirtualLowRankLinear(nn.Module):
    """
    Linear layer W \\in R^{out x in} parameterized as U * V with max_rank columns,
    but only first 'rank' columns actively used.

    - U_full: [out, max_rank] (parameter)
    - V_full: [max_rank, in] (parameter)
    - active rank r = self.rank (<= max_rank)
    - We train entire U_full/V_full with AdamW; changing r does not touch optimizer state.
    - Periodically, we analyze the effective gradient dW and adapt r via SpectralRankController.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int,
        max_rank: int | None = None,
        min_rank: int = 8,
        energy_target: float = 0.99,
        svd_interval_init: int = 500,
        svd_interval_max: int = 4000,
        svd_grow_factor: float = 1.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if max_rank is None:
            max_rank = min(in_features, out_features)
        max_rank = int(max_rank)
        init_rank = int(min(max(init_rank, min_rank), max_rank))

        # Full buffers
        self.U_full = nn.Parameter(
            torch.randn(out_features, max_rank) / math.sqrt(out_features)
        )
        self.V_full = nn.Parameter(
            torch.randn(max_rank, in_features) / math.sqrt(in_features)
        )

        self.controller = SpectralRankController(
            init_rank=init_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_target=energy_target,
            ema_decay=0.8,
            max_step_down=8,
            max_step_up=2,
        )

        self.rank = init_rank
        self.max_rank = max_rank

        # Lazy / adaptive SVD scheduling
        self.svd_interval = svd_interval_init
        self.svd_interval_max = svd_interval_max
        self.svd_grow_factor = svd_grow_factor
        self.next_svd_step = svd_interval_init

    def forward(self, x):
        # x: [B, T, in_features]
        r = self.rank
        U = self.U_full[:, :r]       # [out, r]
        V = self.V_full[:r, :]       # [r, in]
        # y = x @ W^T = x @ (V^T U^T)
        return x @ V.t() @ U.t()

    @torch.no_grad()
    def _gradient_spectrum(self):
        """
        Build approximate full gradient dW from factor grads and compute its spectrum.

        dW ≈ dU * V + U * dV, using active rank slice.
        Returns svals (1D tensor) or None if grads missing.
        """
        if self.U_full.grad is None or self.V_full.grad is None:
            return None

        r = self.rank
        if r <= 0:
            return None

        U = self.U_full[:, :r]          # [out, r]
        V = self.V_full[:r, :]          # [r, in]
        dU = self.U_full.grad[:, :r]    # [out, r]
        dV = self.V_full.grad[:r, :]    # [r, in]

        # Effective gradient of W
        G = dU @ V + U @ dV             # [out, in]

        try:
            svals = torch.linalg.svdvals(G)
        except RuntimeError:
            # Fallback
            U_s, svals, Vh_s = torch.linalg.svd(G, full_matrices=False)
        return svals

    @torch.no_grad()
    def maybe_update_rank(self, global_step: int, logger: JsonLogger | None, layer_name: str):
        """
        Called after backward(), before optimizer.step().
        Uses gradient spectrum occasionally to update the active rank.
        """
        if global_step < self.next_svd_step:
            return

        svals = self._gradient_spectrum()
        if svals is None:
            # No gradient; try again later
            self.next_svd_step = global_step + self.svd_interval
            return

        spectral_rank, tail_energy = self.controller.propose_from_svals(svals)
        old_rank, new_rank = self.controller.update(spectral_rank)

        # Adapt SVD interval: if tail is tiny, we can relax checks (things are stable)
        if tail_energy < 1e-4:
            self.svd_interval = min(
                int(self.svd_interval * self.svd_grow_factor),
                self.svd_interval_max,
            )
        # Else: keep interval as is (you could also shrink it if tail_energy is high)

        self.next_svd_step = global_step + self.svd_interval
        self.rank = new_rank

        if logger is not None and new_rank != old_rank:
            logger.log(
                {
                    "type": "rank_update",
                    "step": global_step,
                    "layer": layer_name,
                    "old_rank": int(old_rank),
                    "new_rank": int(new_rank),
                    "spectral_rank": int(spectral_rank),
                    "tail_energy": float(tail_energy),
                    "svd_interval": int(self.svd_interval),
                    "svals_len": int(svals.numel()),
                }
            )


# ============================================================
# TRANSFORMER BLOCK WITH ADAPTIVE LOW-RANK LINEARS
# ============================================================

class AdaptiveTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        init_rank: int,
        max_rank: int | None = None,
        min_rank: int = 8,
        energy_target: float = 0.99,
        svd_interval_init: int = 500,
        svd_interval_max: int = 4000,
        svd_grow_factor: float = 1.5,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        def LR(in_f, out_f):
            return VirtualLowRankLinear(
                in_features=in_f,
                out_features=out_f,
                init_rank=init_rank,
                max_rank=max_rank,
                min_rank=min_rank,
                energy_target=energy_target,
                svd_interval_init=svd_interval_init,
                svd_interval_max=svd_interval_max,
                svd_grow_factor=svd_grow_factor,
            )

        # Multi-head attention projections
        self.q_proj = LR(d_model, d_model)
        self.k_proj = LR(d_model, d_model)
        self.v_proj = LR(d_model, d_model)
        self.o_proj = LR(d_model, d_model)

        # Feedforward
        ff_dim = ff_mult * d_model
        self.ff1 = LR(d_model, ff_dim)
        self.ff2 = LR(ff_dim, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.lowrank_layers = [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
            ("ff1", self.ff1),
            ("ff2", self.ff2),
        ]

    def attention(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head)

        # scaled dot-product attention
        att_scores = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.d_head)
        att = F.softmax(att_scores, dim=-1)
        out = torch.einsum("bhtT,bThd->bthd", att, v)
        return out.reshape(B, T, C)

    def forward(self, x):
        # Pre-norm block
        x = x + self.attention(self.ln1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x

    @torch.no_grad()
    def maybe_update_ranks(self, global_step: int, logger: JsonLogger | None, prefix: str):
        for name, layer in self.lowrank_layers:
            layer_name = f"{prefix}.{name}"
            layer.maybe_update_rank(global_step, logger, layer_name)


# ============================================================
# FULL TRANSFORMER MODEL
# ============================================================

class AdaptiveLowRankTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        max_rank: int | None = None,
        init_rank: int = 64,
        min_rank: int = 8,
        energy_target: float = 0.99,
        svd_interval_init: int = 500,
        svd_interval_max: int = 4000,
        svd_grow_factor: float = 1.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)  # enough for typical block sizes

        self.blocks = nn.ModuleList(
            [
                AdaptiveTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    init_rank=init_rank,
                    max_rank=max_rank,
                    min_rank=min_rank,
                    energy_target=energy_target,
                    svd_interval_init=svd_interval_init,
                    svd_interval_max=svd_interval_max,
                    svd_grow_factor=svd_grow_factor,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # [1, T]
        x = self.embed(idx) + self.pos_embed(pos)
        for layer in self.blocks:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def maybe_update_all_ranks(self, global_step: int, logger: JsonLogger | None):
        for i, block in enumerate(self.blocks):
            block.maybe_update_ranks(global_step, logger, prefix=f"block{i}")

    @torch.no_grad()
    def collect_rank_stats(self):
        ranks = []
        for block in self.blocks:
            for _, lr in block.lowrank_layers:
                ranks.append(lr.rank)
        if not ranks:
            return 0, 0.0, 0
        ranks = torch.tensor(ranks, dtype=torch.float32)
        return int(ranks.min().item()), float(ranks.mean().item()), int(ranks.max().item())


# ============================================================
# TRAIN LOOP
# ============================================================

def evaluate(model, data, batch_size, block_size):
    model.eval()
    losses = []
    with torch.no_grad():
        # Use a few batches as validation estimate
        for _ in range(50):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            losses.append(loss.item())
    return sum(losses) / len(losses)


def train(
    data_file: str,
    epochs: int,
    init_rank: int,
    log_file: str | None,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    ff_mult: int = 4,
    block_size: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
):

    train_data, val_data, vocab_size = load_token_file(data_file)

    model = AdaptiveLowRankTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        max_rank=init_rank,      # we treat init_rank also as max_rank here
        init_rank=init_rank,
        min_rank=8,
        energy_target=0.99,
        svd_interval_init=500,
        svd_interval_max=4000,
        svd_grow_factor=1.5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    logger = JsonLogger(log_file)

    print("Training on", device)
    print(f"Initial Rank: {init_rank}")

    # Steps per epoch approximately covering the dataset once
    steps_per_epoch = max(1, train_data.size(0) // (batch_size * block_size))
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step in range(steps_per_epoch):
            global_step += 1
            x, y = get_batch(train_data, batch_size, block_size)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Spectral rank updates based on gradient
            model.maybe_update_all_ranks(global_step, logger)

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / steps_per_epoch
        val_loss = evaluate(model, val_data, batch_size, block_size)
        t1 = time.time()

        r_min, r_avg, r_max = model.collect_rank_stats()
        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"R(min/avg/max)={r_min}/{r_avg:.1f}/{r_max} | "
            f"{t1 - t0:.2f}s"
        )

        if logger is not None:
            logger.log(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "rank_min": int(r_min),
                    "rank_avg": float(r_avg),
                    "rank_max": int(r_max),
                    "epoch_time_sec": float(t1 - t0),
                }
            )

    if logger is not None:
        logger.close()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--log-file", type=str, default=None)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    train(
        data_file=args.data_file,
        epochs=args.epochs,
        init_rank=args.init_rank,
        log_file=args.log_file,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
