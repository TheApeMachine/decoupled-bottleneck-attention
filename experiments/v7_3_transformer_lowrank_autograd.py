import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ======================================================
# 1. CUSTOM AUTO-GRAD LOW RANK LAYER
# ======================================================

class LowRankLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, V):
        # x: [B*T, in], U: [out, r], V: [r, in]
        Wt = (U @ V).t()           # [in, out]
        out = x @ Wt               # [B*T, out]
        ctx.save_for_backward(x, U, V)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, U, V = ctx.saved_tensors
        r = V.size(0)

        # grad wrt x
        grad_x = grad_out @ (U @ V)

        # grad wrt U and V (factorized gradients)
        grad_U = grad_out.t() @ (x @ V.t())
        grad_V = (U.t() @ grad_out.t()) @ x

        return grad_x, grad_U, grad_V



# ======================================================
# 2. RANK CONTROLLER (EMA + MOMENTUM)
# ======================================================

class RankController:
    def __init__(
        self,
        init_rank,
        ema_decay=0.9,
        adjust_rate=0.05,
        update_every=2,
        min_rank=8,
        max_rank=256,
        momentum=0.8,
    ):
        self.rank = init_rank
        self.ema_target = float(init_rank)
        self.ema_decay = ema_decay
        self.adjust_rate = adjust_rate
        self.update_every = update_every
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.momentum = momentum

        self.velocity = 0.0
        self.step = 0

    def update(self, suggested):
        # smooth target
        self.ema_target = (
            self.ema_decay * self.ema_target
            + (1 - self.ema_decay) * suggested
        )

        self.step += 1
        if (self.step % self.update_every) != 0:
            return self.rank

        delta = self.ema_target - self.rank
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta
        self.rank += self.velocity * self.adjust_rate

        # clamp & cast
        self.rank = int(max(self.min_rank, min(self.rank, self.max_rank)))
        return self.rank


# ======================================================
# 3. LowRankLinear — uses the custom autograd function
# ======================================================

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank_controller = RankController(rank)

        # U: [out, r], V: [r, in]
        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))

    def forward(self, x):
        B, T, C = x.shape
        x2 = x.reshape(B*T, C)
        out = LowRankLinearFn.apply(x2, self.U, self.V)
        return out.reshape(B, T, self.out_features)

    def suggested_rank(self):
        gU = self.U.grad.norm().item() if self.U.grad is not None else 0.0
        gV = self.V.grad.norm().item() if self.V.grad is not None else 0.0
        g = (gU + gV) / 2

        return max(8, min(16 + int(4*g), 256))

    def apply_rank_update(self):
        new_rank = self.rank_controller.update(self.suggested_rank())
        old_rank = self.U.shape[1]

        if new_rank == old_rank:
            return

        # Recompute SVD projection
        with torch.no_grad():
            W = self.U @ self.V
            Uw, S, Vh = torch.linalg.svd(W, full_matrices=False)

            S = S[:new_rank]
            Uw = Uw[:, :new_rank]
            Vh = Vh[:new_rank, :]

            self.U = nn.Parameter(Uw * S)
            self.V = nn.Parameter(Vh)

        print(f"Adjusted rank {old_rank} → {new_rank} for layer ({self.out_features} x {self.in_features})")



# ======================================================
# 4. Transformer Blocks with low-rank projections
# ======================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, rank=64):
        super().__init__()

        H = n_heads
        d_head = d_model // H
        self.H = H
        self.d_head = d_head

        self.q = LowRankLinear(d_model, d_model, rank)
        self.k = LowRankLinear(d_model, d_model, rank)
        self.v = LowRankLinear(d_model, d_model, rank)
        self.o = LowRankLinear(d_model, d_model, rank)

        self.ff1 = LowRankLinear(d_model, 256, rank)
        self.ff2 = LowRankLinear(256, d_model, rank)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def attention(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.H, self.d_head)
        k = self.k(x).view(B, T, self.H, self.d_head)
        v = self.v(x).view(B, T, self.H, self.d_head)

        att = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        out = torch.einsum("bhtT,bThd->bthd", att, v)
        return out.reshape(B, T, C)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x

    def update_ranks(self):
        for lr in [self.q, self.k, self.v, self.o, self.ff1, self.ff2]:
            lr.apply_rank_update()


# ======================================================
# 5. Full tiny transformer
# ======================================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, rank=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, 4, rank) for _ in range(n_layers)
        ])
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


# ======================================================
# 6. Training loop
# ======================================================

def train_model(epochs=20, init_rank=64):

    text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them?
"""

    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], device=device)
    train = data[:-100]
    val = data[-100:]

    model = TinyTransformer(len(chars), rank=init_rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    B = 16
    T = 32

    def get_batch(split):
        d = train if split == "train" else val
        if len(d) <= T+1:
            raise ValueError("Dataset too small for chosen sequence length.")
        ix = torch.randint(len(d)-T-1, (B,))
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    for ep in range(1, epochs+1):
        model.train()
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        x, y = get_batch("val")
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x).reshape(-1, logits.size(-1)), y.reshape(-1))

        print(f"Epoch {ep:02d} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")

        model.update_all_ranks()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    args = parser.parse_args()

    train_model(epochs=args.epochs, init_rank=args.init_rank)
