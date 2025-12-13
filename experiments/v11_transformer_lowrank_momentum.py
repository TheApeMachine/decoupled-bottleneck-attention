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
# DATA LOADING (character-level, from text file)
# ============================================================

def load_text_dataset(path, train_frac=0.9):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    # Simple char-level vocab
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * train_frac)
    train_data = data[:n]
    val_data = data[n:]
    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    return train_data.to(device), val_data.to(device), vocab_size, stoi, itos

# ============================================================
# 1. SVD HELPERS
# ============================================================

@torch.no_grad()
def truncated_svd_weight(W: torch.Tensor, max_rank: int):
    """
    Perform SVD on W (CPU if needed), truncate to max_rank.
    Return U_factor, V_factor such that W ≈ U_factor @ V_factor
    with rank r <= max_rank.
    """
    device = W.device
    W_cpu = W.detach().cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    r = min(max_rank, S.numel())
    if r == 0:
        # Degenerate, just keep tiny random factorization
        m, n = W.shape
        U_new = torch.zeros(m, 1)
        V_new = torch.zeros(1, n)
        return U_new.to(device), V_new.to(device)

    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    # Distribute sqrt(S) to U and V for better conditioning
    S_sqrt = torch.sqrt(S)
    U_factor = U * S_sqrt.unsqueeze(0)       # [m, r]
    V_factor = S_sqrt.unsqueeze(1) * Vh      # [r, n]

    return U_factor.to(device), V_factor.to(device), S.to(device)


@torch.no_grad()
def truncated_svd_momentum(M: torch.Tensor, max_rank: int):
    """
    Low-rank approximation for momentum M ≈ P @ Q, rank <= max_rank.
    """
    device = M.device
    M_cpu = M.detach().cpu()
    U, S, Vh = torch.linalg.svd(M_cpu, full_matrices=False)
    r = min(max_rank, S.numel())
    if r == 0:
        m, n = M.shape
        P = torch.zeros(m, 1)
        Q = torch.zeros(1, n)
        return P.to(device), Q.to(device)

    U = U[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    S_sqrt = torch.sqrt(S)
    P = U * S_sqrt.unsqueeze(0)       # [m, r]
    Q = S_sqrt.unsqueeze(1) * Vh      # [r, n]

    return P.to(device), Q.to(device)

# ============================================================
# 2. RANK CONTROLLER (spectral energy)
# ============================================================

class SpectralRankController:
    """
    Chooses rank based on explained variance of singular values.
    Keeps an EMA of target rank to avoid jitter.
    """
    def __init__(
        self,
        init_rank: int,
        min_rank: int,
        max_rank: int,
        energy_threshold: float = 0.99,
        ema_decay: float = 0.9,
        adjust_every: int = 1,
    ):
        self.rank = init_rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.energy_threshold = energy_threshold
        self.ema_decay = ema_decay
        self.adjust_every = adjust_every
        self._ema_target = float(init_rank)
        self._step = 0

    @torch.no_grad()
    def update(self, singular_values: torch.Tensor):
        """
        Given singular values S (descending), update desired rank.
        """
        self._step += 1
        if self._step % self.adjust_every != 0:
            return self.rank

        # Explained variance
        s2 = singular_values.pow(2)
        total = s2.sum()
        if total <= 0:
            target = self.min_rank
        else:
            cumsum = torch.cumsum(s2, dim=0)
            frac = cumsum / total
            # smallest k where energy >= threshold
            idx = (frac >= self.energy_threshold).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                target = len(singular_values)
            else:
                target = int(idx[0].item()) + 1  # +1 because idx is 0-based

        target = max(self.min_rank, min(target, self.max_rank))

        # Smooth with EMA
        self._ema_target = self.ema_decay * self._ema_target + (1 - self.ema_decay) * target
        new_rank = int(round(self._ema_target))
        new_rank = max(self.min_rank, min(new_rank, self.max_rank))

        self.rank = new_rank
        return self.rank

# ============================================================
# 3. LOW-RANK LINEAR WITH LOW-RANK MOMENTUM IN W-SPACE
# ============================================================

class LowRankMomentumLinear(nn.Module):
    """
    W ≈ U @ V   (rank r)
    Momentum M ≈ P @ Q  (rank k_mom)
    We:
      - reconstruct G_W from grad(U), grad(V)
      - update momentum in W-space
      - update W in W-space
      - project W back to rank r via SVD (with spectral rank controller)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_rank: int,
        mom_rank: int = 8,
        min_rank: int = 8,
        max_rank: int | None = None,
        energy_threshold: float = 0.99,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if max_rank is None:
            max_rank = min(in_features, out_features)

        init_rank = min(init_rank, max_rank)
        init_rank = max(init_rank, min_rank)

        self.rank_controller = SpectralRankController(
            init_rank=init_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
            ema_decay=0.9,
            adjust_every=1,
        )

        r = init_rank
        # Initial factors
        self.U = nn.Parameter(torch.randn(out_features, r) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(r, in_features) / math.sqrt(in_features))

        # Momentum factors (not registered as parameters – pure optimizer state)
        self.register_buffer("P_mom", None, persistent=False)
        self.register_buffer("Q_mom", None, persistent=False)
        self.mom_rank = mom_rank

    @property
    def current_rank(self):
        return self.U.shape[1]

    def forward(self, x):
        # x: [B, T, in_features] or [..., in_features]
        return x @ self.V.t() @ self.U.t()

    @torch.no_grad()
    def _full_weight(self):
        return self.U @ self.V

    @torch.no_grad()
    def _full_momentum(self):
        if self.P_mom is None or self.Q_mom is None:
            return None
        return self.P_mom @ self.Q_mom

    @torch.no_grad()
    def _approx_full_grad(self):
        """
        Approximate G_W from grad(U), grad(V).

        Heuristic: G_W ≈ dU @ V + U @ dV
        """
        if self.U.grad is None or self.V.grad is None:
            return None

        dU = self.U.grad
        dV = self.V.grad
        G = dU @ self.V + self.U @ dV
        return G

    @torch.no_grad()
    def optim_step(self, lr: float, beta: float = 0.9, eps: float = 1e-8):
        """
        One optimizer step in W-space with low-rank momentum.

        - builds G_W from U.grad, V.grad
        - updates low-rank momentum
        - updates W
        - projects W back to low-rank with spectral rank controller
        """
        G = self._approx_full_grad()
        # Clear grads for this layer; we handle updates manually
        if self.U.grad is not None:
            self.U.grad.zero_()
        if self.V.grad is not None:
            self.V.grad.zero_()

        if G is None:
            return  # nothing to do this step

        # --------- Momentum update in W-space ---------
        M_prev = self._full_momentum()
        if M_prev is None:
            M_full = G
        else:
            M_full = beta * M_prev + (1 - beta) * G

        # Keep momentum low-rank
        P, Q = truncated_svd_momentum(M_full, self.mom_rank)
        self.P_mom = P
        self.Q_mom = Q
        M_full = P @ Q  # reconstructed low-rank momentum

        # --------- Weight update in W-space ----------
        W = self._full_weight()
        # RMS scaling (Adam-ish, but without second moment)
        rms = M_full.pow(2).mean().sqrt()
        scaled_step = M_full / (rms + eps)
        W_updated = W - lr * scaled_step

        # --------- Project back to low rank ----------
        # SVD projection + spectral rank control
        U_factor, V_factor, S = truncated_svd_weight(W_updated, max_rank=self.rank_controller.max_rank)
        # Update rank controller from actual singular values
        new_rank = self.rank_controller.update(S)

        # Truncate to controller rank
        r = min(new_rank, U_factor.shape[1])
        self.U.detach_()
        self.V.detach_()
        self.U = nn.Parameter(U_factor[:, :r])
        self.V = nn.Parameter(V_factor[:r, :])

        # Note: momentum is *not* projected to match W's singular subspace;
        # it's simply kept low-rank in W-space. That's acceptable, since we
        # recompute M_full each step from P, Q in the same coordinate system.

# ============================================================
# 4. TRANSFORMER COMPONENTS
# ============================================================

class LowRankAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        init_rank: int,
        mom_rank: int,
        min_rank: int,
        max_rank: int,
        energy_threshold: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = LowRankMomentumLinear(
            d_model, d_model, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )
        self.k_proj = LowRankMomentumLinear(
            d_model, d_model, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )
        self.v_proj = LowRankMomentumLinear(
            d_model, d_model, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )
        self.o_proj = LowRankMomentumLinear(
            d_model, d_model, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )

        self.ln1 = nn.LayerNorm(d_model)

    def attention(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head)

        # [B, h, T, T]
        scores = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.d_head)
        att = F.softmax(scores, dim=-1)
        out = torch.einsum("bhtT,bThd->bthd", att, v)
        return out.reshape(B, T, C)

    def forward(self, x):
        h = self.ln1(x)
        attn_out = self.attention(h)
        return x + attn_out

    def lowrank_layers(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.o_proj]


class LowRankFFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        init_rank: int,
        mom_rank: int,
        min_rank: int,
        max_rank: int,
        energy_threshold: float = 0.99,
    ):
        super().__init__()
        self.fc1 = LowRankMomentumLinear(
            d_model, d_ff, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )
        self.fc2 = LowRankMomentumLinear(
            d_ff, d_model, init_rank,
            mom_rank=mom_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            energy_threshold=energy_threshold,
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln2(x)
        ff = self.fc2(F.gelu(self.fc1(h)))
        return x + ff

    def lowrank_layers(self):
        return [self.fc1, self.fc2]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        init_rank: int,
        mom_rank: int,
        min_rank: int,
        max_rank: int,
        energy_threshold: float = 0.99,
    ):
        super().__init__()
        self.attn = LowRankAttentionBlock(
            d_model, n_heads, init_rank, mom_rank,
            min_rank, max_rank, energy_threshold
        )
        self.ffn = LowRankFFNBlock(
            d_model, d_ff, init_rank, mom_rank,
            min_rank, max_rank, energy_threshold
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

    def lowrank_layers(self):
        return self.attn.lowrank_layers() + self.ffn.lowrank_layers()


class TinyLowRankTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        init_rank: int = 64,
        mom_rank: int = 8,
        min_rank: int = 8,
        max_rank: int | None = None,
        energy_threshold: float = 0.99,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if max_rank is None:
            max_rank = d_model  # safe upper bound

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff,
                init_rank=init_rank,
                mom_rank=mom_rank,
                min_rank=min_rank,
                max_rank=max_rank,
                energy_threshold=energy_threshold,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        tok = self.token_emb(idx)                    # [B, T, C]
        pos = self.pos_emb(pos)[None, :, :].expand(B, T, -1)
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def lowrank_layers(self):
        layers = []
        for blk in self.blocks:
            layers.extend(blk.lowrank_layers())
        return layers

# ============================================================
# 5. TRAINING LOOP
# ============================================================

def get_batch(data: torch.Tensor, B: int, T: int):
    # data: [N]
    N = data.size(0)
    ix = torch.randint(0, N - T - 1, (B,), device=data.device)
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x, y

def train_model(
    data_file: str,
    log_file: str | None = None,
    epochs: int = 30,
    init_rank: int = 64,
    mom_rank: int = 8,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 1024,
    lr_lowrank: float = 1e-3,
    lr_other: float = 3e-4,
    batch_size: int = 32,
    block_size: int = 128,
    steps_per_epoch: int = 200,
):

    train_data, val_data, vocab_size, stoi, itos = load_text_dataset(data_file)

    model = TinyLowRankTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        init_rank=init_rank,
        mom_rank=mom_rank,
        min_rank=8,
        max_rank=d_model,
        energy_threshold=0.99,
        max_seq_len=block_size,
    ).to(device)

    # Separate non-lowrank params for AdamW
    lowrank_params = set()
    for layer in model.lowrank_layers():
        lowrank_params.add(layer.U)
        lowrank_params.add(layer.V)

    other_params = [p for p in model.parameters() if p not in lowrank_params]

    if other_params:
        opt_other = torch.optim.AdamW(other_params, lr=lr_other)
    else:
        opt_other = None

    print("Training on", device)
    print(f"Vocab size: {vocab_size}")
    print(f"Initial Rank (low-rank layers): {init_rank}")

    log_fh = None
    if log_file is not None:
        log_fh = open(log_file, "w", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, batch_size, block_size)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # Backward
            if opt_other is not None:
                opt_other.zero_grad(set_to_none=True)
            loss.backward()

            # Low-rank momentum step for all low-rank layers
            with torch.no_grad():
                for layer in model.lowrank_layers():
                    layer.optim_step(lr=lr_lowrank, beta=0.9)

            # Update non-lowrank params
            if opt_other is not None:
                opt_other.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / steps_per_epoch

        # Eval
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, batch_size, block_size)
            logits_val = model(x_val)
            val_loss = F.cross_entropy(
                logits_val.view(-1, logits_val.size(-1)),
                y_val.view(-1)
            ).item()

        # Rank stats
        ranks = [layer.current_rank for layer in model.lowrank_layers()]
        min_r = min(ranks)
        max_r = max(ranks)
        avg_r = sum(ranks) / len(ranks)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"R(min/avg/max)={min_r}/{avg_r:.1f}/{max_r} | "
            f"{dt:.2f}s"
        )

        if log_fh is not None:
            rec = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "rank_min": min_r,
                "rank_avg": avg_r,
                "rank_max": max_r,
                "time_sec": dt,
            }
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()

    if log_fh is not None:
        log_fh.close()

# ============================================================
# 6. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to text file")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--mom-rank", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--lr-lowrank", type=float, default=1e-3)
    parser.add_argument("--lr-other", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    args = parser.parse_args()

    train_model(
        data_file=args.data_file,
        log_file=args.log_file,
        epochs=args.epochs,
        init_rank=args.init_rank,
        mom_rank=args.mom_rank,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        lr_lowrank=args.lr_lowrank,
        lr_other=args.lr_other,
        batch_size=args.batch_size,
        block_size=args.block_size,
        steps_per_epoch=args.steps_per_epoch,
    )
