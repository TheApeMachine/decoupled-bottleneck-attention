import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------
# 1. Rank Controller with EMA + momentum
# ------------------------------

class RankController:
    def __init__(
        self,
        init_rank,
        ema_decay=0.9,         # EMA smoothing
        adjust_rate=0.05,      # How fast we move toward EMA target
        update_every=2,        # Only adjust ranks every K epochs
        min_rank=8,
        max_rank=256,
        momentum=0.8,          # Rank momentum dampens oscillations
    ):
        self.rank = init_rank
        self.ema_decay = ema_decay
        self.adjust_rate = adjust_rate
        self.update_every = update_every
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.momentum = momentum

        self.ema_target = float(init_rank)
        self.velocity = 0.0
        self.step_counter = 0

    def update(self, new_suggested_rank):
        # Update EMA (slow-changing target)
        self.ema_target = (
            self.ema_decay * self.ema_target
            + (1 - self.ema_decay) * new_suggested_rank
        )

        self.step_counter += 1
        if self.step_counter % self.update_every != 0:
            return self.rank  # no change yet

        # Compute direction toward target
        delta = self.ema_target - self.rank

        # Momentum update
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta

        # Apply small step
        self.rank += self.adjust_rate * self.velocity

        # Clamp
        self.rank = int(max(self.min_rank, min(self.rank, self.max_rank)))

        return self.rank


# ---------------------------
# Low-rank linear layer
# ---------------------------

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank_controller = RankController(rank)

        # Initial low-rank factors
        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))

    def forward(self, x):
        return x @ self.V.t() @ self.U.t()

    def suggested_rank(self):
        # Gradient norm heuristic: if U/V gradients large → increase rank
        gU = self.U.grad.norm().item() if self.U.grad is not None else 0
        gV = self.V.grad.norm().item() if self.V.grad is not None else 0
        s = (gU + gV) / 2

        # Map norm → rank suggestion (tunable)
        target = 16 + int(4 * s)
        return max(8, min(target, 256))

    def apply_rank_update(self):
        new_rank = self.rank_controller.update(self.suggested_rank())
        if new_rank == self.U.shape[1]:
            return  # no change

        old_rank = self.U.shape[1]

        # Resize using SVD projection (safe)
        with torch.no_grad():
            W = self.U @ self.V
            Uw, S, Vh = torch.linalg.svd(W, full_matrices=False)
            S = S[:new_rank]
            Uw = Uw[:, :new_rank]
            Vh = Vh[:new_rank, :]

            self.U = nn.Parameter(Uw * S)
            self.V = nn.Parameter(Vh)

        print(f"Adjusted rank {old_rank} → {new_rank} for layer ({self.out_features} x {self.in_features})")


# ---------------------------
# Transformer block
# ---------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, rank=64):
        super().__init__()

        self.attn_q = LowRankLinear(d_model, d_model, rank)
        self.attn_k = LowRankLinear(d_model, d_model, rank)
        self.attn_v = LowRankLinear(d_model, d_model, rank)
        self.attn_o = LowRankLinear(d_model, d_model, rank)

        self.ff1 = LowRankLinear(d_model, 256, rank)
        self.ff2 = LowRankLinear(256, d_model, rank)

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

    def update_ranks(self):
        for layer in [
            self.attn_q, self.attn_k, self.attn_v, self.attn_o,
            self.ff1, self.ff2
        ]:
            layer.apply_rank_update()


# ---------------------------
# Transformer model
# ---------------------------

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, rank=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, 4, rank) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.fc_out(x)

    def update_all_ranks(self):
        for b in self.blocks:
            b.update_ranks()


# ---------------------------
# Training loop
# ---------------------------

def train_model(epochs=20, rank=64, lr=3e-4):
    text = """
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
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], device=device)
    train = data[:-100]
    val = data[-100:]

    model = TinyTransformer(len(chars), rank=rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    B = 16
    T = 32

    def get_batch(split):
        d = train if split == "train" else val
        ix = torch.randint(len(d)-T, (B,))
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    for epoch in range(1, epochs+1):
        model.train()
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        x, y = get_batch("val")
        with torch.no_grad():
            v_logits = model(x)
            v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), y.view(-1))

        print(f"Epoch {epoch:02d} | Train {loss.item():.4f} | Val {v_loss.item():.4f}")

        model.update_all_ranks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    args = parser.parse_args()

    train_model(epochs=args.epochs, rank=args.init_rank)
