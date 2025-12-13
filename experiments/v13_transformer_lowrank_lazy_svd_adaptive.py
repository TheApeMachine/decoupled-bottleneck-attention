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
        print("Using MPS backend.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    print("Using CPU backend.")
    return torch.device("cpu")

device = get_device()

# ============================================================
# DATA LOADING (CHAR-LEVEL)
# ============================================================

def load_char_dataset(path, val_fraction=0.1):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = len(data)
    n_val = int(n * val_fraction)
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    print(
        f"Vocab size: {len(chars)}, "
        f"Train tokens: {len(train_data)}, "
        f"Val tokens: {len(val_data)}"
    )
    return train_data, val_data, stoi, itos

# ============================================================
# ADAPTIVE LOW-RANK LINEAR (FIXED MAX RANK + LAZY SVD)
# ============================================================

# Global SVD / rank-control hyperparams (can be tweaked)
SVD_TARGET_ENERGY   = 0.98   # how much spectral energy we want to keep
SVD_TAIL_LOW        = 0.01   # if tail energy below this -> increase interval
SVD_TAIL_HIGH       = 0.05   # if tail energy above this -> decrease interval
SVD_INIT_INTERVAL   = 200    # initial steps between SVDs
SVD_MIN_INTERVAL    = 20
SVD_MAX_INTERVAL    = 1000
SVD_GROWTH_FACTOR   = 1.5    # how much interval grows when system is "calm"
SVD_DECAY_FACTOR    = 0.7    # how much interval shrinks when rank is mis-set
RANK_STEP_MAX       = 4      # max rank delta per SVD call
RANK_MIN            = 8      # absolute lower bound
WARMUP_STEPS        = 500    # don't touch ranks before this many steps

class LowRankLinearAdaptive(nn.Module):
    """
    Core idea:
    - We allocate U_full: [out, max_rank], V_full: [max_rank, in]
    - We maintain an integer self.rank, and only use the first self.rank columns/rows.
    - This means parameter *shapes never change*. So AdamW keeps all its momentum state.
    - Rank changes are just integer pointer changes; SVD only used to *measure* spectrum.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int,
        max_rank: int = None,
        target_energy: float = SVD_TARGET_ENERGY,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if max_rank is None:
            max_rank = init_rank
        max_rank = min(max_rank, in_features, out_features)

        self.max_rank = max_rank
        self.rank = min(init_rank, max_rank)
        self.target_energy = target_energy

        # Fixed-size factors
        scale = 1.0 / math.sqrt(out_features)
        self.U_full = nn.Parameter(torch.randn(out_features, max_rank) * scale)
        self.V_full = nn.Parameter(torch.randn(max_rank, in_features) * scale)

        # SVD scheduling state
        self.last_svd_step = 0
        self.svd_interval = SVD_INIT_INTERVAL

    def forward(self, x):
        # x: [B, T, in_features] or [*, in_features]
        U = self.U_full[:, :self.rank]        # [out, r]
        V = self.V_full[:self.rank, :]        # [r, in]
        return x @ V.t() @ U.t()

    @torch.no_grad()
    def maybe_svd_update(self, global_step: int):
        """
        Lazy + adaptive SVD:
        - Only runs if enough steps elapsed since last SVD.
        - Only runs after warmup.
        - Uses spectrum of W = U_active @ V_active to:
            * Suggest new rank
            * Adjust how often the next SVD should be run
        Importantly, we DO NOT change U_full / V_full values here,
        only the rank and the SVD interval. So optimizer state is untouched.
        """
        if global_step < WARMUP_STEPS:
            return

        steps_since = global_step - self.last_svd_step
        if steps_since < self.svd_interval:
            return

        # Build effective W for current active rank
        U = self.U_full[:, :self.rank]   # [out, r]
        V = self.V_full[:self.rank, :]   # [r, in]
        W = U @ V                        # [out, in]

        # Run SVD on CPU if needed (MPS fallback)
        W_cpu = W.detach().to("cpu")
        try:
            U_svd, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        except RuntimeError as e:
            # If SVD fails for some reason, just skip this round
            print(f"[WARN] SVD failed in LowRankLinearAdaptive: {e}")
            self.last_svd_step = global_step
            return

        # Energy profile
        S2 = S.pow(2)
        total_energy = S2.sum().item()
        if total_energy <= 0:
            # Degenerate; just mark and bail out
            self.last_svd_step = global_step
            return

        cum_energy = torch.cumsum(S2, dim=0)
        energy_ratio = cum_energy / total_energy  # [min(out,in)]

        # Spectral rank suggestion = smallest k s.t. cumulative energy >= target_energy
        spectral_rank = int((energy_ratio >= self.target_energy).nonzero(as_tuple=True)[0][0].item() + 1)
        spectral_rank = max(RANK_MIN, min(spectral_rank, self.max_rank))

        # Tail energy given CURRENT rank (how much we're throwing away)
        current_rank = min(self.rank, len(energy_ratio))
        tail_energy = 1.0 - energy_ratio[current_rank - 1].item()

        # Smooth rank change: don't jump wildly
        new_rank = self.rank
        if spectral_rank < self.rank:
            # want to shrink
            new_rank = max(spectral_rank, self.rank - RANK_STEP_MAX)
        elif spectral_rank > self.rank:
            # want to grow
            new_rank = min(spectral_rank, self.rank + RANK_STEP_MAX)

        # Clamp rank
        new_rank = max(RANK_MIN, min(new_rank, self.max_rank))

        # Update rank pointer
        old_rank = self.rank
        self.rank = new_rank

        # Adaptive SVD interval logic
        interval = self.svd_interval
        if tail_energy < SVD_TAIL_LOW:
            # We are "over-parameterized" for this layer; spectrum is very concentrated
            interval = min(int(interval * SVD_GROWTH_FACTOR), SVD_MAX_INTERVAL)
        elif tail_energy > SVD_TAIL_HIGH:
            # We are cutting off too much energy with current rank
            interval = max(int(interval * SVD_DECAY_FACTOR), SVD_MIN_INTERVAL)
        # else: leave interval unchanged

        self.svd_interval = max(SVD_MIN_INTERVAL, min(interval, SVD_MAX_INTERVAL))
        self.last_svd_step = global_step

        # Optional: debug print (can be commented out for silence)
        if old_rank != new_rank:
            print(
                f"[SVD] step={global_step} "
                f"({self.out_features}x{self.in_features}) "
                f"rank {old_rank} -> {new_rank} | "
                f"spectral={spectral_rank} | tail_energy={tail_energy:.4f} | "
                f"next_interval={self.svd_interval}"
            )

    def get_rank(self):
        return self.rank

# ============================================================
# TRANSFORMER BLOCK WITH LOW-RANK ATTENTION + MLP
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, init_rank, max_rank_mult=2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        max_rank = min(d_model, init_rank * max_rank_mult)

        self.q_proj = LowRankLinearAdaptive(d_model, d_model, init_rank, max_rank)
        self.k_proj = LowRankLinearAdaptive(d_model, d_model, init_rank, max_rank)
        self.v_proj = LowRankLinearAdaptive(d_model, d_model, init_rank, max_rank)
        self.o_proj = LowRankLinearAdaptive(d_model, d_model, init_rank, max_rank)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1, 1, 1024, 1024))  # will slice to needed T at runtime
        )

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]

        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # [B, H, T, T]

        # causal mask
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

    def lowrank_modules(self):
        return [
            self.q_proj, self.k_proj, self.v_proj, self.o_proj
        ]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, init_rank, max_rank_mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = CausalSelfAttention(d_model, n_heads, init_rank, max_rank_mult)

        max_rank_ff = min(d_model, init_rank * max_rank_mult)
        self.ff1 = LowRankLinearAdaptive(d_model, d_ff, init_rank, max_rank_ff)
        self.ff2 = LowRankLinearAdaptive(d_ff, d_model, init_rank, max_rank_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x

    def lowrank_modules(self):
        return self.attn.lowrank_modules() + [self.ff1, self.ff2]

class LowRankTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=None,
        init_rank=64,
        max_rank_mult=2,
        max_seq_len=512,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, init_rank, max_rank_mult)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # collect low-rank modules once for easy rank updates
        self._lowrank_modules = []
        for m in self.modules():
            if isinstance(m, LowRankLinearAdaptive):
                self._lowrank_modules.append(m)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.size()
        assert T <= self.max_seq_len

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # [1, T]

        x = self.tok_emb(idx) + self.pos_emb(pos)  # [B, T, C]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def update_all_ranks(self, global_step: int):
        for m in self._lowrank_modules:
            m.maybe_svd_update(global_step)

    @torch.no_grad()
    def rank_stats(self):
        ranks = [m.get_rank() for m in self._lowrank_modules]
        if not ranks:
            return 0, 0.0, 0
        r_min = int(min(ranks))
        r_max = int(max(ranks))
        r_avg = float(sum(ranks) / len(ranks))
        return r_min, r_avg, r_max

# ============================================================
# TRAINING LOOP
# ============================================================

def get_batch(data, batch_size, seq_len):
    # data: 1D LongTensor
    n = data.size(0)
    # random starting positions
    ix = torch.randint(0, n - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def estimate_loss(model, train_data, val_data, seq_len, batch_size, eval_steps=50):
    model.eval()
    losses = {}
    with torch.no_grad():
        for split, data in [("train", train_data), ("val", val_data)]:
            total = 0.0
            for _ in range(eval_steps):
                x, y = get_batch(data, batch_size, seq_len)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                total += loss.item()
            losses[split] = total / eval_steps
    model.train()
    return losses["train"], losses["val"]

def train(
    data_file,
    epochs=30,
    init_rank=64,
    d_model=256,
    n_layers=6,
    n_heads=4,
    max_rank_mult=2,
    seq_len=256,
    batch_size=32,
    lr=3e-4,
    steps_per_epoch=200,
    log_file=None,
):
    train_data, val_data, stoi, itos = load_char_dataset(data_file)

    model = LowRankTransformer(
        vocab_size=len(stoi),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        init_rank=init_rank,
        max_rank_mult=max_rank_mult,
        max_seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    global_step = 0

    log_fp = None
    if log_file is not None:
        log_fp = open(log_file, "w", encoding="utf-8")

    print("Training on", device)
    print(f"Initial Rank: {init_rank}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, batch_size, seq_len)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            # lazy, adaptive rank updates
            model.update_all_ranks(global_step)

        # periodic eval
        train_loss, val_loss = estimate_loss(
            model, train_data, val_data, seq_len, batch_size, eval_steps=50
        )
        r_min, r_avg, r_max = model.rank_stats()
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"R(min/avg/max)={r_min}/{r_avg:.1f}/{r_max} | "
            f"{dt:.2f}s"
        )

        # JSONL logging
        if log_fp is not None:
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "rank_min": r_min,
                "rank_avg": r_avg,
                "rank_max": r_max,
                "epoch_time_sec": dt,
            }
            log_fp.write(json.dumps(rec) + "\n")
            log_fp.flush()

    if log_fp is not None:
        log_fp.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-rank-mult", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--log-file", type=str, default=None)

    args = parser.parse_args()

    train(
        data_file=args.data_file,
        epochs=args.epochs,
        init_rank=args.init_rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_rank_mult=args.max_rank_mult,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        log_file=args.log_file,
    )
