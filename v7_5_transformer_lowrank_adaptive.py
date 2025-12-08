import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

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
# EFFECTIVE RANK (Spectral)
# ============================================================

@torch.no_grad()
def effective_rank(W, energy_threshold=0.90):
    """
    Computes smallest rank r that captures `energy_threshold`
    of spectral energy of W.
    """
    W_cpu = W.detach().float().cpu()
    s = torch.linalg.svdvals(W_cpu)

    total = s.sum().item()
    running = 0.0
    r = 0

    for sv in s:
        running += sv.item()
        r += 1
        if running / total >= energy_threshold:
            break

    return max(1, r)

# ============================================================
# RANK CONTROLLER
# ============================================================

class RankController:
    def __init__(
        self,
        init_rank,
        ema_decay=0.9,
        adjust_rate=0.2,
        update_every=1,
        min_rank=8,
        max_rank=256,
        momentum=0.8,
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
        self.ema_target = (
            self.ema_decay * self.ema_target
            + (1 - self.ema_decay) * new_suggested_rank
        )

        self.step_counter += 1
        if self.step_counter % self.update_every != 0:
            return self.rank

        delta = self.ema_target - self.rank
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta
        self.rank += self.adjust_rate * self.velocity

        self.rank = int(max(self.min_rank, min(self.rank, self.max_rank)))

        if abs(self.rank - self.ema_target) < 1.0:
            self.rank = int(round(self.ema_target))

        return self.rank

# ============================================================
# RANDOMIZED SVD RESIZE
# ============================================================

@torch.no_grad()
def randomized_svd_resize(U, V, new_rank):
    W = U @ V
    m, n = W.shape
    W_cpu = W.detach().cpu()

    try:
        Uw, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        Uw = Uw[:, :new_rank]
        S = S[:new_rank]
        Vh = Vh[:new_rank, :]

        S_sqrt = torch.diag(torch.sqrt(S))
        U_new = Uw @ S_sqrt
        V_new = S_sqrt @ Vh

        return U_new.to(U.device), V_new.to(U.device)

    except Exception as e:
        print(f"SVD failed: {e}")
        return (
            torch.randn(m, new_rank, device=U.device) * 0.02,
            torch.randn(new_rank, n, device=U.device) * 0.02,
        )

# ============================================================
# LOW-RANK LINEAR LAYER
# ============================================================

class SympatheticLowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank_controller = RankController(
            rank,
            ema_decay=0.9,
            adjust_rate=0.2,
            min_rank=8,
            max_rank=min(in_features, out_features),
        )

        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))

    def forward(self, x):
        return x @ self.V.t() @ self.U.t()

    def get_suggested_rank(self):
        with torch.no_grad():
            W = self.U @ self.V
            r = effective_rank(W, energy_threshold=0.90)
            return r

    def apply_rank_update(self):
        suggested = self.get_suggested_rank()
        new_rank = self.rank_controller.update(suggested)
        current_rank = self.U.shape[1]

        if new_rank == current_rank:
            return False

        new_U, new_V = randomized_svd_resize(self.U.data, self.V.data, new_rank)
        self.U = nn.Parameter(new_U)
        self.V = nn.Parameter(new_V)

        print(f"Layer ({self.out_features}x{self.in_features}) rank: {current_rank} -> {new_rank}")
        return True

# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, rank=64):
        super().__init__()

        self.attn_q = SympatheticLowRankLinear(d_model, d_model, rank)
        self.attn_k = SympatheticLowRankLinear(d_model, d_model, rank)
        self.attn_v = SympatheticLowRankLinear(d_model, d_model, rank)
        self.attn_o = SympatheticLowRankLinear(d_model, d_model, rank)

        self.ff1 = SympatheticLowRankLinear(d_model, 256, rank)
        self.ff2 = SympatheticLowRankLinear(256, d_model, rank)

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
        changed = False
        layers = [
            self.attn_q, self.attn_k, self.attn_v, self.attn_o,
            self.ff1, self.ff2
        ]
        for layer in layers:
            if layer.apply_rank_update():
                changed = True
        return changed

# ============================================================
# FULL TRANSFORMER
# ============================================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=3, rank=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, 4, rank) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.fc_out(x)

    def update_all_ranks(self):
        changed = False
        for b in self.blocks:
            if b.update_ranks():
                changed = True
        return changed

# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(epochs=20, rank=64, lr=3e-4):
    data = torch.tensor([stoi[c] for c in tiny_text], device=device)
    train = data[:-100]
    val = data[-100:]

    model = TinyTransformer(len(chars), rank=rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    B = 4
    T = 16

    def get_batch(split):
        d = train if split == "train" else val
        if len(d) <= T:
            return d[:-1].unsqueeze(0), d[1:].unsqueeze(0)

        ix = torch.randint(len(d)-T, (B,))
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    print(f"Training on {device}...")
    print(f"Initial Rank: {rank}")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for _ in range(10):
            x, y = get_batch("train")
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / 10

        # Eval
        model.eval()
        x, y = get_batch("val")
        with torch.no_grad():
            v_logits = model(x)
            v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), y.view(-1))

        print(f"Epoch {epoch:02d} | Train {avg_loss:.4f} | Val {v_loss.item():.4f}")

        # Rank update schedule
        if epoch > 4 and epoch % 3 == 0:
            if model.update_all_ranks():
                opt = torch.optim.AdamW(model.parameters(), lr=lr)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    args = parser.parse_args()

    train_model(epochs=args.epochs, rank=args.init_rank)
