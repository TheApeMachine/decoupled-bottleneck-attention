import math
import time
import json
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
# DATA LOADING (WORD-LEVEL, WIKITEXT-FRIENDLY)
# ============================================================

def load_token_dataset(path, val_fraction=0.01):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tokens = text.strip().split()

    # vocab
    vocab = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}

    data = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)
    n = int(len(data) * (1.0 - val_fraction))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocab size: {len(vocab)}, Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    return train_data, val_data, stoi, itos

# ============================================================
# RANK CONTROLLER (SPECTRAL / BIDIRECTIONAL)
# ============================================================

class RankController:
    def __init__(
        self,
        init_rank,
        min_rank,
        max_rank,
        ema_decay=0.8,
        safety_factor=1.5,
        change_threshold=2.0,
        max_step=4,
    ):
        self.rank = int(init_rank)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.ema_decay = ema_decay
        self.safety_factor = safety_factor
        self.change_threshold = change_threshold
        self.max_step = max_step
        self.ema_target = float(init_rank)

    def propose(self, intrinsic_rank):
        # Scale intrinsic rank up a bit to avoid brutal compression
        target = intrinsic_rank * self.safety_factor
        target = max(self.min_rank, min(target, self.max_rank))

        # EMA smoothing on target
        self.ema_target = (
            self.ema_decay * self.ema_target
            + (1.0 - self.ema_decay) * target
        )

        delta = self.ema_target - self.rank
        if abs(delta) < self.change_threshold:
            return self.rank  # no change

        # Move gradually toward EMA target
        step = int(max(-self.max_step, min(self.max_step, round(delta))))
        new_rank = self.rank + step
        new_rank = max(self.min_rank, min(new_rank, self.max_rank))
        return int(new_rank)

    def commit(self, new_rank):
        self.rank = int(new_rank)


# ============================================================
# LOW-RANK LINEAR LAYER WITH SPECTRAL RANK ESTIMATION
# ============================================================

class LowRankLinearAdaptiveSpectral(nn.Module):
    def __init__(self, in_features, out_features, init_rank, min_rank=8, max_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if max_rank is None:
            max_rank = min(in_features, out_features)

        init_rank = int(min(max(init_rank, min_rank), max_rank))

        self.controller = RankController(
            init_rank=init_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            ema_decay=0.8,
            safety_factor=1.5,
            change_threshold=2.0,
            max_step=4,
        )

        r = init_rank
        # U: [out, r], V: [r, in]
        self.U = nn.Parameter(torch.randn(out_features, r) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(r, in_features) / math.sqrt(in_features))

    @property
    def rank(self):
        return self.U.shape[1]

    def forward(self, x):
        # x: [..., in_features]
        # W = U @ V  =>  y = x @ W^T = x @ V^T @ U^T
        return x @ self.V.t() @ self.U.t()

    @torch.no_grad()
    def estimate_intrinsic_rank(self):
        """
        Estimate an 'intrinsic' rank using stable rank:
            r_eff = ||W||_F^2 / sigma_max(W)^2
        where sigma_max is approximated via a short power iteration.
        """
        eps = 1e-8
        U = self.U
        V = self.V

        W = U @ V  # [out, in]
        fro2 = (W * W).sum()

        # Power iteration for top singular value
        _, in_dim = W.shape
        v = torch.randn(in_dim, device=W.device)
        for _ in range(5):
            v = v / (v.norm() + eps)
            u = W @ v
            u = u / (u.norm() + eps)
            v = W.t() @ u

        v = v / (v.norm() + eps)
        u = (W @ v)
        sigma = (u.norm() + eps)  # approx top singular value

        stable_rank = (fro2 / (sigma * sigma + eps)).item()
        # Clamp to valid range
        stable_rank = max(1.0, min(stable_rank, min(self.out_features, self.in_features)))
        return stable_rank

    @torch.no_grad()
    def _resize_to_rank(self, new_rank):
        current_rank = self.rank
        if new_rank == current_rank:
            return False

        new_rank = int(new_rank)
        new_rank = max(1, min(new_rank, min(self.out_features, self.in_features)))

        # Use SVD on CPU for stability and to avoid MPS limitations
        W = (self.U @ self.V).detach().cpu()
        try:
            Uw, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(new_rank, S.shape[0])
            Uw = Uw[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]

            S_sqrt = torch.sqrt(S + 1e-8)
            U_new = Uw * S_sqrt.unsqueeze(0)      # [out, k]
            V_new = S_sqrt.unsqueeze(1) * Vh      # [k, in]
        except Exception as e:
            print(f"[WARN] SVD failed ({e}), falling back to random init for rank {new_rank}")
            U_new = torch.randn(self.out_features, new_rank) * 0.02
            V_new = torch.randn(new_rank, self.in_features) * 0.02

        U_new = U_new.to(self.U.device)
        V_new = V_new.to(self.V.device)

        self.U = nn.Parameter(U_new)
        self.V = nn.Parameter(V_new)
        return True

    @torch.no_grad()
    def update_rank(self):
        intrinsic = self.estimate_intrinsic_rank()
        proposed = self.controller.propose(intrinsic)
        changed = self._resize_to_rank(proposed)
        if changed:
            self.controller.commit(self.rank)
        return changed, intrinsic


# ============================================================
# TRANSFORMER BLOCK (GPT-STYLE)
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, init_rank=64, min_rank=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Attention projections
        self.q_proj = LowRankLinearAdaptiveSpectral(d_model, d_model, init_rank, min_rank=min_rank)
        self.k_proj = LowRankLinearAdaptiveSpectral(d_model, d_model, init_rank, min_rank=min_rank)
        self.v_proj = LowRankLinearAdaptiveSpectral(d_model, d_model, init_rank, min_rank=min_rank)
        self.o_proj = LowRankLinearAdaptiveSpectral(d_model, d_model, init_rank, min_rank=min_rank)

        # FFN
        self.ffn_up = LowRankLinearAdaptiveSpectral(d_model, d_ff, init_rank, min_rank=min_rank)
        self.ffn_down = LowRankLinearAdaptiveSpectral(d_ff, d_model, init_rank, min_rank=min_rank)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def self_attention(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d]
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, T, T]
        att = F.softmax(att, dim=-1)
        out = att @ v  # [B, h, T, d]

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.o_proj(out)
        return out

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffn_down(F.gelu(self.ffn_up(self.ln2(x))))
        return x

    @torch.no_grad()
    def update_ranks(self):
        changed = False
        for mod in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.ffn_up,
            self.ffn_down,
        ]:
            c, _ = mod.update_rank()
            changed = changed or c
        return changed

    def collect_ranks(self):
        ranks = []
        for mod in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.ffn_up,
            self.ffn_down,
        ]:
            ranks.append(mod.rank)
        return ranks


# ============================================================
# GPT MODEL WITH ADAPTIVE LOW-RANK LAYERS
# ============================================================

class GPTLowRank(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        block_size=256,
        init_rank=64,
        min_rank=8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                init_rank=init_rank,
                min_rank=min_rank,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # [1, T]

        x = self.token_emb(idx) + self.pos_emb(pos)  # [B, T, C]

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def update_all_ranks(self):
        any_changed = False
        for blk in self.blocks:
            c = blk.update_ranks()
            any_changed = any_changed or c
        return any_changed

    @torch.no_grad()
    def collect_rank_stats(self):
        ranks = []
        for blk in self.blocks:
            ranks.extend(blk.collect_ranks())
        if not ranks:
            return None
        ranks = torch.tensor(ranks, dtype=torch.float32)
        return {
            "min": int(ranks.min().item()),
            "max": int(ranks.max().item()),
            "avg": float(ranks.mean().item()),
            "all": [int(x) for x in ranks.tolist()],
        }


# ============================================================
# TRAINING LOOP
# ============================================================

def train(
    data_file,
    log_file=None,
    epochs=30,
    init_rank=64,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    block_size=256,
    batch_size=32,
    steps_per_epoch=200,
    lr=3e-4,
    min_rank=8,
):
    train_data, val_data, stoi, itos = load_token_dataset(data_file)

    vocab_size = len(stoi)
    model = GPTLowRank(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        block_size=block_size,
        init_rank=init_rank,
        min_rank=min_rank,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    def get_batch(split):
        source = train_data if split == "train" else val_data
        if len(source) <= block_size:
            raise RuntimeError("Data too short for given block size")
        ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
        x = torch.stack([source[i:i+block_size] for i in ix])
        y = torch.stack([source[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    log_f = None
    if log_file is not None:
        log_f = open(log_file, "w", encoding="utf-8")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Training on {device}")
    print(f"Initial Rank: {init_rank}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = get_batch("train")
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / steps_per_epoch

        # Eval
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            val_logits = model(x_val)
            val_loss = F.cross_entropy(
                val_logits.view(-1, val_logits.size(-1)),
                y_val.view(-1),
            ).item()

        # Rank update (once per epoch)
        with torch.no_grad():
            changed = model.update_all_ranks()
            stats = model.collect_rank_stats()

        # If ranks changed, rebuild optimizer to avoid stale state
        if changed:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        dt = time.time() - t0

        if stats is None:
            rmin = rmax = ravg = 0
        else:
            rmin, rmax, ravg = stats["min"], stats["max"], stats["avg"]

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"R(min/avg/max)={rmin}/{ravg:.1f}/{rmax} | "
            f"{dt:.2f}s"
        )

        if log_f is not None and stats is not None:
            rec = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "rank_min": stats["min"],
                "rank_max": stats["max"],
                "rank_avg": stats["avg"],
                "ranks": stats["all"],
                "params_millions": total_params / 1e6,
                "time_sec": dt,
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    if log_f is not None:
        log_f.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to tokenized text file (space-separated)")
    parser.add_argument("--log-file", type=str, default=None, help="JSONL log output path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-rank", type=int, default=8)

    args = parser.parse_args()

    train(
        data_file=args.data_file,
        log_file=args.log_file,
        epochs=args.epochs,
        init_rank=args.init_rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        block_size=args.block_size,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        min_rank=args.min_rank,
    )
