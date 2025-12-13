#!/usr/bin/env python3
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
    elif torch.cuda.is_available():
        print("Using CUDA backend.")
        return torch.device("cuda")
    print("Using CPU backend.")
    return torch.device("cpu")

device = get_device()

# ============================================================
# DATA LOADING
# ============================================================

def load_data(path: str, val_fraction: float = 0.1, tokenizer_model: str = "gpt2"):
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Check if it looks like space-separated integers (pre-tokenized)
    # We look at the first 50 tokens to guess
    sample_tokens = text.strip().split(maxsplit=50)
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
        ids = [int(t) for t in text.strip().split()]
        data = torch.tensor(ids, dtype=torch.long)
        vocab_size = int(data.max().item()) + 1
        print(f"Vocab size (from max token id): {vocab_size}")
    else:
        print(f"Detected raw text. Using tiktoken ({tokenizer_model})...")
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_model)
            ids = enc.encode(text, allowed_special={'<|endoftext|>'})
            data = torch.tensor(ids, dtype=torch.long)
            vocab_size = enc.n_vocab
            print(f"Vocab size (tiktoken): {vocab_size}")
        except ImportError:
            print("\n[ERROR] 'tiktoken' not found. Please install it: pip install tiktoken")
            print("Falling back to simple character-level tokenization.")
            chars = sorted(list(set(text)))
            stoi = {c: i for i, c in enumerate(chars)}
            data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
            vocab_size = len(chars)
            print(f"Vocab size (char-level): {vocab_size}")

    n = int(len(data) * (1.0 - val_fraction))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    return train_data.to(device), val_data.to(device), vocab_size

def get_batch(data, batch_size: int, seq_len: int):
    # Random crop
    L = data.size(0)
    ix = torch.randint(0, L - seq_len - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ============================================================
# VIRTUAL LOW RANK LINEAR (v13-style)
# ============================================================

class VirtualLowRankLinear(nn.Module):
    """
    Low-rank linear with:
      - Fixed maximum rank buffers (U_full, V_full)
      - Active rank via slicing (no shape changes -> optimizer state preserved)
      - Lazy SVD-based spectral rank control with adaptive interval
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int,
        init_rank: int = None,
        svd_init_interval: int = 500,
        svd_growth: float = 1.35,
        svd_tail_target: float = 1e-3,
        min_rank: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.rank = init_rank if init_rank is not None else max_rank
        self.min_rank = min_rank

        # Parameters: full buffers
        self.U_full = nn.Parameter(
            torch.randn(out_features, max_rank) / math.sqrt(out_features)
        )
        self.V_full = nn.Parameter(
            torch.randn(max_rank, in_features) / math.sqrt(in_features)
        )

        # SVD scheduling
        self.svd_step = 0
        self.svd_interval = svd_init_interval
        self.svd_growth = svd_growth
        self.svd_tail_target = svd_tail_target
        self.svd_max_interval = 5000

        # Last SVD stats (for logging / debugging)
        self.last_spectral_rank = self.rank
        self.last_tail_energy = None

    def forward(self, x):
        r = self.rank
        U = self.U_full[:, :r]
        V = self.V_full[:r, :]
        return x @ V.t() @ U.t()

    @torch.no_grad()
    def _run_svd_update(self):
        r = self.rank
        U = self.U_full[:, :r]
        V = self.V_full[:r, :]

        # Reconstruct active weight
        W = U @ V  # [out, in]

        # SVD on CPU for stability on MPS
        W_cpu = W.detach().cpu()
        try:
            U_s, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        except Exception as e:
            print(f"[SVD] failed on ({self.out_features}x{self.in_features}): {e}")
            return

        # Compute spectral rank based on cumulative energy
        S2 = S * S
        total = S2.sum()
        if total <= 0:
            spectral_rank = self.min_rank
            tail_energy = 1.0
        else:
            cumsum = torch.cumsum(S2, dim=0)
            frac = cumsum / total
            spectral_rank = int((frac < (1.0 - self.svd_tail_target)).sum().item()) + 1
            spectral_rank = max(self.min_rank, min(spectral_rank, self.max_rank))
            tail_energy = float(1.0 - frac[spectral_rank - 1].item())

        # New target rank is spectral_rank, but don't jump more than, say, 8 at once
        RANK_STEP_MAX = 8
        if spectral_rank > self.rank:
            new_rank = min(self.rank + RANK_STEP_MAX, spectral_rank)
        else:
            new_rank = max(self.rank - RANK_STEP_MAX, spectral_rank)

        new_rank = max(self.min_rank, min(new_rank, self.max_rank))

        # Truncate factors to new_rank
        U_s = U_s[:, :new_rank]          # [out, new_rank]
        S_trunc = S[:new_rank]
        Vh = Vh[:new_rank, :]            # [new_rank, in]

        S_sqrt = torch.sqrt(S_trunc)
        # Distribute sqrt(S) symmetrically
        U_new = U_s * S_sqrt.unsqueeze(0)
        V_new = S_sqrt.unsqueeze(1) * Vh

        # Write back into full buffers (in-place, no reallocation)
        self.U_full.data[:, :new_rank].copy_(U_new.to(self.U_full.device))
        self.V_full.data[:new_rank, :].copy_(V_new.to(self.V_full.device))

        # Optionally keep some of the old columns beyond new_rank as "ghost" capacity
        # (we leave them untouched)

        # Update rank and stats
        old_rank = self.rank
        self.rank = new_rank
        self.last_spectral_rank = spectral_rank
        self.last_tail_energy = tail_energy

        print(
            f"[SVD] step={self.svd_step} "
            f"({self.out_features}x{self.in_features}) "
            f"rank {old_rank} -> {new_rank} | "
            f"spectral={spectral_rank} | tail_energy={tail_energy:.4g} | "
            f"next_interval={self.svd_interval}"
        )

        # Adaptive interval: if we are very low-tail, check less often
        if tail_energy < self.svd_tail_target * 0.3:
            self.svd_interval = min(
                int(self.svd_interval * self.svd_growth),
                self.svd_max_interval,
            )

    def maybe_svd_update(self):
        if not self.training:
            return
        self.svd_step += 1
        if self.svd_step >= self.svd_interval:
            self._run_svd_update()
            # Reset step counter and schedule next check
            self.svd_step = 0

# ============================================================
# MULTI-HEAD ATTENTION WITH ADAPTIVE HEAD GATING
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using VirtualLowRankLinear for Q/K/V/O and
    adaptive head gating based on spectral energy of per-head slices.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_rank: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Low-rank projections
        self.q_proj = VirtualLowRankLinear(d_model, d_model, max_rank=max_rank)
        self.k_proj = VirtualLowRankLinear(d_model, d_model, max_rank=max_rank)
        self.v_proj = VirtualLowRankLinear(d_model, d_model, max_rank=max_rank)
        self.o_proj = VirtualLowRankLinear(d_model, d_model, max_rank=max_rank)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Head mask and scores
        self.register_buffer("head_mask", torch.ones(n_heads, dtype=torch.bool))
        self.register_buffer("head_scores", torch.zeros(n_heads))

        # Head update schedule (lazy, like SVD)
        self.head_step = 0
        self.head_interval = 500
        self.head_max_interval = 5000
        self.head_growth = 1.3

        # Hysteresis thresholds (normalized 0..1)
        self.head_threshold_off = 0.03
        self.head_threshold_on = 0.06
        self.min_active_heads = 2

    def _compute_head_scores_from_weights(self):
        """
        Spectral-ish energy per head from Q/K/V weights using Frobenius norm
        of per-head slices of W = U @ V.

        For each proj in {Q,K,V}:
          - precompute B = V V^T
          - for each head:
                U_h = U[head_rows]
                A = U_h^T U_h
                energy = trace(A B) = ||U_h V||_F^2 = sum sigma_i^2
        Sum energies across Q,K,V.

        We NEVER build full W on large matrices; all ops are in factor space.
        """
        scores = torch.zeros(self.n_heads, device=self.head_scores.device)

        with torch.no_grad():
            projs = [self.q_proj, self.k_proj, self.v_proj]
            for proj in projs:
                r = proj.rank
                U = proj.U_full[:, :r]       # [d_model, r]
                V = proj.V_full[:r, :]       # [r, d_model]

                # B is shared across heads for this projection
                B = V @ V.t()                # [r, r]

                for h in range(self.n_heads):
                    start = h * self.d_head
                    end = (h + 1) * self.d_head
                    U_h = U[start:end, :]    # [d_head, r]
                    A = U_h.t() @ U_h        # [r, r]
                    # Frobenius^2 of W_h = trace(A B)
                    energy = torch.trace(A @ B)
                    scores[h] += energy

        # Clamp to avoid tiny negative due to numerical issues
        scores.clamp_(min=0.0)
        return scores

    @torch.no_grad()
    def _update_head_mask(self):
        scores = self._compute_head_scores_from_weights()
        self.head_scores.copy_(scores)

        max_score = float(scores.max().item())
        if max_score <= 0.0:
            # Degenerate; keep mask as-is
            return

        norm_scores = scores / max_score

        # Start from current mask and apply hysteresis
        new_mask = self.head_mask.clone()

        # Turn OFF heads that are active but very low energy
        off_candidates = (self.head_mask) & (norm_scores < self.head_threshold_off)
        new_mask[off_candidates] = False

        # Turn ON heads that are inactive but clearly above ON threshold
        on_candidates = (~self.head_mask) & (norm_scores > self.head_threshold_on)
        new_mask[on_candidates] = True

        # Enforce minimum active heads
        if new_mask.sum() < self.min_active_heads:
            # Activate top-k heads by score
            topk = torch.topk(norm_scores, k=self.min_active_heads, largest=True).indices
            new_mask[topk] = True

        changed = (new_mask != self.head_mask).sum().item()
        self.head_mask.copy_(new_mask)

        # Adaptive interval: if heads are very concentrated, check less often
        # Concentration = 1 - (mean / max)
        concentration = 1.0 - float(norm_scores.mean().item())
        if concentration < 0.2:
            self.head_interval = min(
                int(self.head_interval * self.head_growth),
                self.head_max_interval,
            )

        # Light logging
        n_active = int(new_mask.sum().item())
        print(
            f"[HEADS] step={self.head_step} "
            f"active={n_active}/{self.n_heads} "
            f"changed={changed} "
            f"max_score={max_score:.4g}"
        )

    def maybe_update_heads(self):
        if not self.training:
            return
        self.head_step += 1
        if self.head_step >= self.head_interval:
            self._update_head_mask()
            self.head_step = 0

    def forward(self, x):
        # x: [B, T, d_model]
        B, T, C = x.shape

        # Update rank in projections (lazy SVD)
        self.q_proj.maybe_svd_update()
        self.k_proj.maybe_svd_update()
        self.v_proj.maybe_svd_update()
        self.o_proj.maybe_svd_update()

        # Possibly update head gating
        self.maybe_update_heads()

        q = self.q_proj(x)   # [B, T, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads
        q = q.view(B, T, self.n_heads, self.d_head)   # [B, T, H, Dh]
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # Apply head mask: masked heads are zeroed out entirely
        # Expand mask to [1, 1, H, 1] to broadcast
        mask = self.head_mask.view(1, 1, self.n_heads, 1).to(q.dtype)
        q = q * mask
        k = k * mask
        v = v * mask

        # Attention scores
        # scores: [B, H, T, T]
        q_ = q.permute(0, 2, 1, 3)  # [B, H, T, Dh]
        k_ = k.permute(0, 2, 1, 3)
        v_ = v.permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        att = F.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        out = torch.matmul(att, v_)  # [B, H, T, Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.proj_dropout(out)
        return out

# ============================================================
# FEEDFORWARD BLOCK
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, max_rank: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc1 = VirtualLowRankLinear(d_model, hidden_dim, max_rank=max_rank)
        self.fc2 = VirtualLowRankLinear(hidden_dim, d_model, max_rank=max_rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.fc1.maybe_svd_update()
        self.fc2.maybe_svd_update()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_rank: int,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_rank=max_rank,
            attn_dropout=attn_dropout,
            proj_dropout=resid_dropout,
        )
        self.ff = FeedForward(d_model, hidden_mult=ffn_mult, max_rank=max_rank, dropout=resid_dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ============================================================
# FULL MODEL
# ============================================================

class AdaptiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_rank: int = 64,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_rank=max_rank,
                ffn_mult=ffn_mult,
                attn_dropout=0.0,
                resid_dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        assert T <= self.max_seq_len, "Sequence length exceeds model max_seq_len"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # [1, T]
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ============================================================
# TRAINING LOOP
# ============================================================

def compute_rank_stats(model: nn.Module):
    ranks = []
    for m in model.modules():
        if isinstance(m, VirtualLowRankLinear):
            ranks.append(m.rank)
    if not ranks:
        return None
    ranks = torch.tensor(ranks, dtype=torch.float)
    return int(ranks.min().item()), float(ranks.mean().item()), int(ranks.max().item())

def compute_head_stats(model: nn.Module):
    heads = []
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            heads.append(int(m.head_mask.sum().item()))
    if not heads:
        return None
    heads = torch.tensor(heads, dtype=torch.float)
    return int(heads.min().item()), float(heads.mean().item()), int(heads.max().item())

def estimate_loss(model, train_data, val_data, batch_size, seq_len, eval_iters=50):
    model.eval()
    losses = {}
    with torch.no_grad():
        for split, data in [("train", train_data), ("val", val_data)]:
            total = 0.0
            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, seq_len)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                total += loss.item()
            losses[split] = total / eval_iters
    model.train()
    return losses

def train(
    data_file: str,
    epochs: int = 30,
    init_rank: int = 64,
    max_rank: int = 64,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 4,
    ffn_mult: int = 4,
    batch_size: int = 32,
    seq_len: int = 256,
    steps_per_epoch: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    log_file: str = None,
    tokenizer: str = "gpt2",
):
    train_data, val_data, vocab_size = load_data(data_file, tokenizer_model=tokenizer)

    print("Training on", device)
    print(
        f"Model: d_model={d_model}, layers={n_layers}, heads={n_heads}, "
        f"max_rank={max_rank}, init_rank={init_rank}"
    )

    model = AdaptiveTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_rank=max_rank,
        ffn_mult=ffn_mult,
        max_seq_len=seq_len,
    ).to(device)

    # Initialize all ranks to init_rank
    for m in model.modules():
        if isinstance(m, VirtualLowRankLinear):
            m.rank = min(init_rank, m.max_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_f = None
    if log_file is not None:
        log_f = open(log_file, "w", encoding="utf-8")

    global_step = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, batch_size, seq_len)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        avg_train_loss = total_loss / steps_per_epoch
        losses = estimate_loss(
            model,
            train_data,
            val_data,
            batch_size=batch_size,
            seq_len=seq_len,
            eval_iters=50,
        )

        rank_stats = compute_rank_stats(model)
        head_stats = compute_head_stats(model)

        epoch_time = time.time() - start_time

        if rank_stats is not None:
            r_min, r_avg, r_max = rank_stats
            r_str = f"R(min/avg/max)={r_min}/{r_avg:.1f}/{r_max}"
        else:
            r_str = "R(n/a)"

        if head_stats is not None:
            h_min, h_avg, h_max = head_stats
            h_str = f"H(min/avg/max)={h_min}/{h_avg:.1f}/{h_max}"
        else:
            h_str = "H(n/a)"

        print(
            f"Epoch {epoch:02d} | "
            f"Train {avg_train_loss:.4f} | Val {losses['val']:.4f} | "
            f"{r_str} | {h_str} | {epoch_time:.2f}s"
        )

        # Logging to JSONL
        if log_f is not None:
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": avg_train_loss,
                "val_loss": losses["val"],
                "rank_min": rank_stats[0] if rank_stats else None,
                "rank_avg": rank_stats[1] if rank_stats else None,
                "rank_max": rank_stats[2] if rank_stats else None,
                "heads_min": head_stats[0] if head_stats else None,
                "heads_avg": head_stats[1] if head_stats else None,
                "heads_max": head_stats[2] if head_stats else None,
                "epoch_time": epoch_time,
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    if log_f is not None:
        log_f.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="tiktoken encoding (gpt2, cl100k_base, etc)")

    args = parser.parse_args()

    train(
        data_file=args.data_file,
        epochs=args.epochs,
        init_rank=args.init_rank,
        max_rank=args.max_rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_file=args.log_file,
        tokenizer=args.tokenizer,
    )
