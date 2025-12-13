import math
import time
import json
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    print("Using CPU backend.")
    return torch.device("cpu")

device = get_device()

# ============================================================
# DATASET
# ============================================================

# Fallback tiny text if no file is provided
FALLBACK_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them? To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
"""

def load_text(data_file: str | None) -> str:
    if data_file is not None:
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text) < 1000:
            print(f"Warning: data file is quite small ({len(text)} chars).")
        return text
    else:
        print("No --data-file provided, using small fallback sample.")
        return FALLBACK_TEXT

def build_char_dataset(text: str, val_fraction: float = 0.1):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(len(data) * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    if len(train_data) < 1024:
        print(f"Warning: train set is small ({len(train_data)} tokens). "
              f"Consider using a larger corpus.")

    return train_data, val_data, vocab_size, stoi, itos

# ============================================================
# RANK CONTROLLER (BIDIRECTIONAL, LOSS-AWARE)
# ============================================================

class RankController:
    def __init__(
        self,
        init_rank: int,
        min_rank: int,
        max_rank: int,
        ema_decay: float = 0.9,
        adjust_rate: float = 0.5,   # how aggressively we move toward EMA
        boost_step: int = 4,        # how much to grow on loss spike
        max_step: int = 4           # max rank change per update
    ):
        self.rank = init_rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.ema_decay = ema_decay
        self.adjust_rate = adjust_rate
        self.boost_step = boost_step
        self.max_step = max_step
        self.ema_target = float(init_rank)

    def update(self, spectral_target: int, loss_trend: int) -> int:
        """
        spectral_target: rank needed to capture target spectral energy.
        loss_trend: -1 (improving), 0 (flat), +1 (worse).
        """
        # Adjust target based on loss trend
        raw_target = spectral_target

        if loss_trend > 0:
            # Validation loss got worse recently → allow some rebound
            raw_target = max(spectral_target, self.rank + self.boost_step)

        # EMA smoothing toward raw_target
        self.ema_target = (
            self.ema_decay * self.ema_target +
            (1.0 - self.ema_decay) * float(raw_target)
        )

        delta = self.ema_target - float(self.rank)

        # Limit how much we move per update
        delta = max(-self.max_step, min(self.max_step, delta))

        # Apply update
        new_rank = int(round(self.rank + self.adjust_rate * delta))

        # Clamp
        new_rank = max(self.min_rank, min(new_rank, self.max_rank))

        # Avoid tiny oscillations around the same integer
        if abs(new_rank - self.rank) <= 0:
            return self.rank

        self.rank = new_rank
        return self.rank

# ============================================================
# SPECTRAL HELPERS
# ============================================================

@torch.no_grad()
def spectral_rank(W: torch.Tensor, energy_target: float, max_rank: int | None = None):
    """
    Given weight matrix W [out, in], compute the smallest r such that
    sum_{i<=r} s_i^2 / sum s_i^2 >= energy_target.
    Returns (rank_r, energy_captured).
    """
    device = W.device
    W_cpu = W.detach().float().cpu()
    try:
        # full_matrices=False gives min(m, n) singular values
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    except Exception as e:
        print(f"SVD failed in spectral_rank: {e}")
        # Fall back to 'full rank'
        r = min(W.shape[0], W.shape[1]) if max_rank is None else max_rank
        return r, 1.0

    energy = S**2
    total = energy.sum().item()
    if total == 0.0:
        r_full = min(W.shape[0], W.shape[1])
        if max_rank is not None:
            r_full = min(r_full, max_rank)
        return r_full, 0.0

    cum = torch.cumsum(energy, dim=0)
    frac = cum / total

    # index where frac >= energy_target
    r = int((frac >= energy_target).nonzero(as_tuple=True)[0][0].item() + 1)
    if max_rank is not None:
        r = min(r, max_rank)

    energy_captured = frac[r-1].item()
    return r, energy_captured

@torch.no_grad()
def spectral_truncate(W: torch.Tensor, new_rank: int):
    """
    Truncate W [out, in] to rank new_rank using SVD and return factors U_new, V_new such that
    W_new ≈ U_new @ V_new.
    """
    device = W.device
    W_cpu = W.detach().float().cpu()
    try:
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    except Exception as e:
        print(f"SVD failed in spectral_truncate: {e}")
        out_dim, in_dim = W.shape
        return (
            torch.randn(out_dim, new_rank, device=device) * 0.02,
            torch.randn(new_rank, in_dim, device=device) * 0.02,
        )

    U = U[:, :new_rank]
    S = S[:new_rank]
    Vh = Vh[:new_rank, :]

    # Distribute sqrt(S) to keep U/V well scaled
    S_sqrt = torch.diag(torch.sqrt(S))
    U_new = U @ S_sqrt
    V_new = S_sqrt @ Vh
    return U_new.to(device), V_new.to(device)

# ============================================================
# LOW RANK LINEAR LAYER (SPECTRAL + CONTROLLER)
# ============================================================

class SpectralLowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, init_rank, min_rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        max_rank = min(in_features, out_features)
        init_rank = max(min_rank, min(init_rank, max_rank))

        self.U = nn.Parameter(
            torch.randn(out_features, init_rank) / math.sqrt(out_features)
        )
        self.V = nn.Parameter(
            torch.randn(init_rank, in_features) / math.sqrt(in_features)
        )

        self.rank_controller = RankController(
            init_rank=init_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            ema_decay=0.9,
            adjust_rate=0.5,
            boost_step=4,
            max_step=4,
        )

    def forward(self, x):
        # x: [B, T, in]
        # V: [r, in], U: [out, r]
        return x @ self.V.t() @ self.U.t()

    def current_rank(self):
        return self.U.shape[1]

    @torch.no_grad()
    def maybe_update_rank(
        self,
        loss_trend: int,
        energy_target: float,
        log_fh,
        epoch: int,
        layer_name: str,
    ):
        """
        Perform SVD-based rank suggestion, feed into controller, and
        resize factors if rank changes.
        """
        # Full weight
        W = self.U @ self.V  # [out, in]

        # Spectral rank estimate
        spec_rank, energy_captured = spectral_rank(
            W,
            energy_target=energy_target,
            max_rank=min(self.in_features, self.out_features),
        )

        old_rank = self.current_rank()
        new_rank = self.rank_controller.update(spec_rank, loss_trend)

        if new_rank != old_rank:
            U_new, V_new = spectral_truncate(W, new_rank)
            self.U = nn.Parameter(U_new)
            self.V = nn.Parameter(V_new)

        # Logging
        if log_fh is not None:
            rec = {
                "epoch": epoch,
                "layer": layer_name,
                "shape": [self.out_features, self.in_features],
                "old_rank": old_rank,
                "new_rank": new_rank,
                "spectral_target": spec_rank,
                "energy_captured": energy_captured,
                "loss_trend": loss_trend,
            }
            log_fh.write(json.dumps(rec) + "\n")

# ============================================================
# TRANSFORMER BLOCK & MODEL
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, init_rank=64, min_rank=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.attn_q = SpectralLowRankLinear(d_model, d_model, init_rank, min_rank)
        self.attn_k = SpectralLowRankLinear(d_model, d_model, init_rank, min_rank)
        self.attn_v = SpectralLowRankLinear(d_model, d_model, init_rank, min_rank)
        self.attn_o = SpectralLowRankLinear(d_model, d_model, init_rank, min_rank)

        self.ff1 = SpectralLowRankLinear(d_model, 4 * d_model, init_rank, min_rank)
        self.ff2 = SpectralLowRankLinear(4 * d_model, d_model, init_rank, min_rank)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def attention(self, x):
        B, T, C = x.shape
        q = self.attn_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.attn_k(x).view(B, T, self.n_heads, self.d_head)
        v = self.attn_v(x).view(B, T, self.n_heads, self.d_head)

        scores = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.d_head)
        att = F.softmax(scores, dim=-1)
        out = torch.einsum("bhtT,bThd->bthd", att, v)
        return out.reshape(B, T, C)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, init_rank=64, min_rank=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, 4, init_rank, min_rank) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)  # [B, T, d]
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.fc_out(x)

    @torch.no_grad()
    def update_all_ranks(self, loss_trend, energy_target, log_fh, epoch):
        # Walk all submodules and update any SpectralLowRankLinear
        for name, module in self.named_modules():
            if isinstance(module, SpectralLowRankLinear):
                module.maybe_update_rank(
                    loss_trend=loss_trend,
                    energy_target=energy_target,
                    log_fh=log_fh,
                    epoch=epoch,
                    layer_name=name,
                )

    @torch.no_grad()
    def rank_stats(self):
        ranks = []
        for module in self.modules():
            if isinstance(module, SpectralLowRankLinear):
                ranks.append(module.current_rank())
        if not ranks:
            return 0, 0.0, 0
        rmin = min(ranks)
        rmax = max(ranks)
        ravg = sum(ranks) / len(ranks)
        return rmin, ravg, rmax

# ============================================================
# LOSS TREND HELPER
# ============================================================

def compute_loss_trend(history, window=3, tol=0.05):
    """
    Look at moving windows of validation loss.
    Returns:
        +1 = recent losses are worse (> tol fraction)
         0 = roughly flat
        -1 = recent losses improved (< -tol fraction)
    """
    if len(history) < 2 * window:
        return 0
    prev = sum(history[-2*window:-window]) / window
    curr = sum(history[-window:]) / window
    if curr > prev * (1 + tol):
        return 1
    if curr < prev * (1 - tol):
        return -1
    return 0

# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    epochs=20,
    init_rank=64,
    d_model=128,
    n_layers=3,
    lr=3e-4,
    batch_size=32,
    block_size=128,
    data_file=None,
    log_file=None,
    energy_target=0.98,
    rank_adjust_every=2,
    warmup_epochs=4,
):
    # Load data
    raw_text = load_text(data_file)
    train_data_cpu, val_data_cpu, vocab_size, stoi, itos = build_char_dataset(raw_text)

    train_data = train_data_cpu.to(device)
    val_data = val_data_cpu.to(device)

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        init_rank=init_rank,
        min_rank=8,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Logging
    log_fh = open(log_file, "w") if log_file is not None else None

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        if data.size(0) <= block_size + 1:
            # just one batch over full sequence
            x = data[:-1].unsqueeze(0)
            y = data[1:].unsqueeze(0)
            return x, y
        idx = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
        return x, y

    val_history = []

    print("Training on", device)
    print(f"Vocab size: {vocab_size}, Train tokens: {train_data.size(0)}, Val tokens: {val_data.size(0)}")
    print(f"Initial Rank: {init_rank}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # A few batches per epoch (this is a toy loop; you can scale it later)
        train_losses = []
        for _ in range(50):
            x, y = get_batch("train")
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            logits_val = model(x_val)
            val_loss = F.cross_entropy(
                logits_val.view(-1, logits_val.size(-1)), y_val.view(-1)
            ).item()

        val_history.append(val_loss)

        # Loss trend & rank update
        loss_trend = compute_loss_trend(val_history, window=3, tol=0.07)

        if epoch >= warmup_epochs and (epoch % rank_adjust_every == 0):
            model.update_all_ranks(
                loss_trend=loss_trend,
                energy_target=energy_target,
                log_fh=log_fh,
                epoch=epoch,
            )
            # optimizer keeps working on same params; we didn't change the object identities,
            # only their data via nn.Parameter re-assignment.

        rmin, ravg, rmax = model.rank_stats()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"R(min/avg/max)={rmin}/{ravg:.1f}/{rmax} | "
            f"{elapsed:.2f}s"
        )

        # Simple JSONL epoch-level record
        if log_fh is not None:
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_trend": loss_trend,
                "rank_min": rmin,
                "rank_avg": ravg,
                "rank_max": rmax,
                "elapsed_sec": elapsed,
            }
            log_fh.write(json.dumps(rec) + "\n")

    if log_fh is not None:
        log_fh.close()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to a large .txt corpus")
    parser.add_argument("--log-file", type=str, default="v9_log.jsonl")
    parser.add_argument("--energy-target", type=float, default=0.98)
    parser.add_argument("--rank-adjust-every", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=4)

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        init_rank=args.init_rank,
        d_model=args.d_model,
        n_layers=args.layers,
        lr=args.lr,
        batch_size=args.batch_size,
        block_size=args.block_size,
        data_file=args.data_file,
        log_file=args.log_file,
        energy_target=args.energy_target,
        rank_adjust_every=args.rank_adjust_every,
        warmup_epochs=args.warmup_epochs,
    )
