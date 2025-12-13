import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

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
# DATASET (tiny Shakespeare fragment)
# ============================================================

tiny_text = """
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

chars = sorted(list(set(tiny_text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)

# ============================================================
# LOGGING (JSONL)
# ============================================================

LOG_PATH = "v8_log.jsonl"

def reset_log():
    with open(LOG_PATH, "w") as f:
        f.write("")

def log_event(obj):
    obj = dict(obj)  # shallow copy
    obj.setdefault("time", time.time())
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(obj) + "\n")

# ============================================================
# LOW-RANK LINEAR WITH SPECTRAL RANK ADAPTATION
# ============================================================

class SpectralLowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, init_rank, name=""):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name or f"lr_{out_features}x{in_features}"

        self.min_rank = 8
        self.max_rank = min(in_features, out_features)
        init_rank = max(self.min_rank, min(self.max_rank, init_rank))
        self.rank = init_rank

        # Factors: W ≈ U @ V, where U:[out, r], V:[r, in]
        self.U = nn.Parameter(
            torch.randn(out_features, init_rank) / math.sqrt(out_features)
        )
        self.V = nn.Parameter(
            torch.randn(init_rank, in_features) / math.sqrt(in_features)
        )

    def forward(self, x):
        # x: [B, T, in_features]
        # W = U @ V -> [out, in]
        # y = x @ W^T = x @ (V^T @ U^T)
        return x @ self.V.t() @ self.U.t()

    @torch.no_grad()
    def update_rank_from_spectrum(
        self,
        epoch: int,
        energy_target: float = 0.95,
        max_step: int = 4,
    ):
        """
        Inspect W = U @ V via SVD, choose a target rank that keeps 'energy_target'
        fraction of spectral energy, and move current rank toward it (limited step).
        Returns (old_rank, new_rank, target_rank, energy_at_new_rank).
        """
        device = self.U.device

        # Reconstruct current effective weight
        W = self.U @ self.V  # [out, in]
        W_cpu = W.detach().cpu()

        try:
            # SVD on CPU (small matrices, this is cheap enough)
            # W = U_svd @ diag(S) @ Vh
            U_svd, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        except Exception as e:
            log_event({
                "type": "svd_error",
                "layer": self.name,
                "epoch": epoch,
                "error": str(e),
            })
            # If SVD fails, do nothing
            return self.rank, self.rank, self.rank, None

        # Spectral energy (proportional to squared singular values)
        energy = S ** 2
        total_energy = energy.sum().item()

        if total_energy <= 0.0:
            # Degenerate, don't touch the rank
            return self.rank, self.rank, self.rank, None

        cum_energy = torch.cumsum(energy, dim=0) / total_energy

        # Find smallest rank that captures desired energy
        idx = (cum_energy >= energy_target).nonzero(as_tuple=False)
        if idx.numel() == 0:
            target_rank = len(S)
        else:
            target_rank = idx[0].item() + 1  # +1 because indices are 0-based

        target_rank = max(self.min_rank, min(self.max_rank, target_rank))

        old_rank = self.rank
        if target_rank == old_rank:
            # No pressure to change, but compute energy at current rank
            energy_at_current = energy[:old_rank].sum().item() / total_energy
            return old_rank, old_rank, target_rank, energy_at_current

        # Limit how much we move per adjustment step
        delta = target_rank - old_rank
        if abs(delta) > max_step:
            new_rank = old_rank + max_step * (1 if delta > 0 else -1)
        else:
            new_rank = target_rank

        new_rank = max(self.min_rank, min(self.max_rank, new_rank))

        if new_rank == old_rank:
            energy_at_current = energy[:old_rank].sum().item() / total_energy
            return old_rank, old_rank, target_rank, energy_at_current

        # Truncate SVD to new_rank and rebuild factors
        U_trunc = U_svd[:, :new_rank]      # [out, new_rank]
        S_trunc = S[:new_rank]             # [new_rank]
        Vh_trunc = Vh[:new_rank, :]        # [new_rank, in]

        # Distribute singular values between U and V (sqrt split)
        S_sqrt = torch.sqrt(S_trunc)       # [new_rank]
        U_new = U_trunc * S_sqrt.unsqueeze(0)   # broadcast over rows
        V_new = S_sqrt.unsqueeze(1) * Vh_trunc  # broadcast over cols

        self.U = nn.Parameter(U_new.to(device))
        self.V = nn.Parameter(V_new.to(device))
        self.rank = int(new_rank)

        energy_at_new = energy[:new_rank].sum().item() / total_energy

        # Log detail to JSONL (no console spam)
        log_event({
            "type": "rank_update",
            "epoch": epoch,
            "layer": self.name,
            "old_rank": int(old_rank),
            "new_rank": int(new_rank),
            "target_rank": int(target_rank),
            "min_rank": int(self.min_rank),
            "max_rank": int(self.max_rank),
            "energy_target": float(energy_target),
            "energy_at_new": float(energy_at_new),
            "total_energy": float(total_energy),
            # small spectrum snapshot
            "top_singular_values": [float(x) for x in S[:10].tolist()],
        })

        return old_rank, int(new_rank), int(target_rank), float(energy_at_new)

# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, init_rank=64, block_id=0):
        super().__init__()

        # Attention projections (all low-rank)
        self.attn_q = SpectralLowRankLinear(d_model, d_model, init_rank, name=f"block{block_id}.attn_q")
        self.attn_k = SpectralLowRankLinear(d_model, d_model, init_rank, name=f"block{block_id}.attn_k")
        self.attn_v = SpectralLowRankLinear(d_model, d_model, init_rank, name=f"block{block_id}.attn_v")
        self.attn_o = SpectralLowRankLinear(d_model, d_model, init_rank, name=f"block{block_id}.attn_o")

        # Feed-forward
        self.ff1 = SpectralLowRankLinear(d_model, 256, init_rank, name=f"block{block_id}.ff1")
        self.ff2 = SpectralLowRankLinear(256, d_model, init_rank, name=f"block{block_id}.ff2")

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

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
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x

    def lowrank_layers(self):
        return [
            self.attn_q, self.attn_k, self.attn_v, self.attn_o,
            self.ff1, self.ff2
        ]

# ============================================================
# TINY TRANSFORMER
# ============================================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, init_rank=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, 4, init_rank, block_id=i)
            for i in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Cache low-rank layers for easy traversal
        self._lr_layers = []
        for b in self.blocks:
            self._lr_layers.extend(b.lowrank_layers())

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.fc_out(x)

    def lowrank_layers(self):
        return self._lr_layers

    def rank_stats(self):
        ranks = [layer.rank for layer in self._lr_layers]
        if not ranks:
            return 0, 0, 0.0
        r_min = min(ranks)
        r_max = max(ranks)
        r_avg = sum(ranks) / len(ranks)
        return r_min, r_max, r_avg

    @torch.no_grad()
    def adjust_ranks(
        self,
        epoch: int,
        energy_target: float = 0.95,
        max_step: int = 4
    ):
        """
        Ask each low-rank layer to adjust its rank based on spectrum.
        Returns (any_changed, r_min, r_max, r_avg).
        """
        any_changed = False
        for layer in self._lr_layers:
            old_rank, new_rank, target_rank, energy_at_new = layer.update_rank_from_spectrum(
                epoch=epoch,
                energy_target=energy_target,
                max_step=max_step,
            )
            if new_rank != old_rank:
                any_changed = True

        r_min, r_max, r_avg = self.rank_stats()
        log_event({
            "type": "rank_adjustment_epoch",
            "epoch": epoch,
            "r_min": int(r_min),
            "r_max": int(r_max),
            "r_avg": float(r_avg),
            "energy_target": float(energy_target),
        })
        return any_changed, r_min, r_max, r_avg

# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    epochs=20,
    init_rank=64,
    lr=3e-4,
    adjust_every=3,
    warmup_epochs=4,
    energy_target=0.95,
):
    reset_log()

    data = torch.tensor([stoi[c] for c in tiny_text], device=device)
    train = data[:-100]
    val = data[-100:]

    model = TinyTransformer(vocab_size, init_rank=init_rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    B = 4
    T = 32

    def get_batch(split):
        d = train if split == "train" else val
        if len(d) <= T + 1:
            x = d[:-1].unsqueeze(0)
            y = d[1:].unsqueeze(0)
            return x, y

        ix = torch.randint(len(d) - T - 1, (B,))
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    print(f"Training on {device}...")
    print(f"Initial Rank: {init_rank}")
    r_min, r_max, r_avg = model.rank_stats()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()

        # A few steps per epoch (tiny dataset)
        train_loss_acc = 0.0
        train_steps = 20

        for _ in range(train_steps):
            x, y = get_batch("train")
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_acc += loss.item()

        train_loss = train_loss_acc / train_steps

        # Validation
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            logits_val = model(x_val)
            val_loss = F.cross_entropy(
                logits_val.view(-1, logits_val.size(-1)),
                y_val.view(-1),
            ).item()

        # Rank adjustment (infrequent, after warmup)
        adjusted_this_epoch = False
        if epoch >= warmup_epochs and (epoch % adjust_every == 0):
            changed, r_min, r_max, r_avg = model.adjust_ranks(
                epoch=epoch,
                energy_target=energy_target,
                max_step=4,
            )
            adjusted_this_epoch = changed
            if changed:
                # Rebuild optimizer because parameter shapes have changed
                opt = torch.optim.AdamW(model.parameters(), lr=lr)

        else:
            r_min, r_max, r_avg = model.rank_stats()

        epoch_time = time.time() - start_time

        # Console: clean, summary-only
        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"R(min/avg/max)={int(r_min)}/{r_avg:.1f}/{int(r_max)} | "
            f"{epoch_time:.2f}s"
        )

        # Log epoch summary
        log_event({
            "type": "epoch_summary",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "r_min": int(r_min),
            "r_max": int(r_max),
            "r_avg": float(r_avg),
            "adjusted_ranks": bool(adjusted_this_epoch),
            "epoch_time": float(epoch_time),
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--adjust-every", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--energy-target", type=float, default=0.95)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        init_rank=args.init_rank,
        lr=args.lr,
        adjust_every=args.adjust_every,
        warmup_epochs=args.warmup_epochs,
        energy_target=args.energy_target,
    )
