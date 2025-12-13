import math
import time
import argparse
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


# ============================================================
# TINY CHARACTER DATASET (YOUR UPDATED TEXT)
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
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s], dtype=torch.long)

def decode(idx_list):
    return ''.join(idx_to_char[i] for i in idx_list)

data = encode(tiny_text)
train_data = data[:-100]
val_data = data[-100:]


# ============================================================
# RANDOMIZED LOW-RANK FACTORIZATION (GPU-FRIENDLY + MPS-SAFE)
# ============================================================

@torch.no_grad()
def randomized_factorization(
    W: torch.Tensor,
    rank: int,
    oversample: int = 8,
    n_iter: int = 1,
):
    """
    Approximate W (m x n) as U @ V with U (m x rank), V (rank x n)
    using a randomized SVD-style factorization.

    Heavy ops (matmuls, QR) run on the current device (MPS/GPU),
    with a small SVD on CPU.
    """
    device = W.device
    m, n = W.shape
    k = min(rank + oversample, min(m, n))

    # Random projection
    Omega = torch.randn(n, k, device=device)
    Y = W @ Omega  # [m, k]

    # Optional power iterations to sharpen spectrum
    for _ in range(n_iter):
        Y = W @ (W.t() @ Y)

    # Orthonormal basis for the subspace
    if Y.device.type == "mps":
        Q_cpu, _ = torch.linalg.qr(Y.cpu(), mode="reduced")
        Q = Q_cpu.to(Y.device)
    else:
        Q, _ = torch.linalg.qr(Y, mode="reduced")  # [m, k]

    # Small core matrix
    B = Q.t() @ W  # [k, n]

    # Move small matrix to CPU for SVD
    B_cpu = B.detach().cpu()
    U_hat, S, Vt = torch.linalg.svd(B_cpu, full_matrices=False)

    # Truncate to desired rank
    U_hat = U_hat[:, :rank]     # [k, rank]
    S = S[:rank]                # [rank]
    Vt = Vt[:rank, :]           # [rank, n]

    # Move back to device
    U_hat = U_hat.to(device)
    S = S.to(device)
    Vt = Vt.to(device)

    # Lift back to full space
    U_tmp = Q @ U_hat           # [m, rank]
    SR = torch.diag(torch.sqrt(S))  # [rank, rank]
    U_new = U_tmp @ SR               # [m, rank]
    V_new = SR @ Vt                  # [rank, n]

    return U_new, V_new


# ============================================================
# CUSTOM LOW-RANK LINEAR AUTOGRAD FUNCTION
# ============================================================

class LowRankLinearFn(torch.autograd.Function):
    """
    Forward: y = x @ V^T @ U^T
    Backward: gradients only through low-rank factors U, V and input x.
    """

    @staticmethod
    def forward(ctx, x, U, V):
        # x: [batch*T, in_dim]
        # U: [out_dim, rank]
        # V: [rank, in_dim]
        h = x @ V.t()          # [N, rank]
        y = h @ U.t()          # [N, out_dim]
        ctx.save_for_backward(x, h, U, V)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [N, out_dim]
        x, h, U, V = ctx.saved_tensors

        # dL/dU = grad_out^T @ h          [out_dim, rank]
        grad_U = grad_out.t() @ h

        # grad_h = grad_out @ U          [N, rank]
        grad_h = grad_out @ U

        # dL/dV = grad_h^T @ x           [rank, in_dim]
        grad_V = grad_h.t() @ x

        # dL/dx = grad_h @ V             [N, in_dim]
        grad_x = grad_h @ V

        return grad_x, grad_U, grad_V


# ============================================================
# ADAPTIVE LOW-RANK LAYER
# ============================================================

class AdaptiveLowRankLayer(nn.Module):
    """
    Represents a weight matrix W (out_dim x in_dim) as U @ V
    with dynamic rank control and a custom low-rank autograd.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, in_dim) * 0.02)

    def forward(self, x):
        # Accepts [..., in_dim], flattens to 2D for autograd, then restores shape.
        orig_shape = x.shape
        in_dim = orig_shape[-1]
        x_flat = x.view(-1, in_dim)
        y_flat = LowRankLinearFn.apply(x_flat, self.U, self.V)
        out = y_flat.view(*orig_shape[:-1], self.out_dim)
        return out

    def reconstruct_full_weight(self):
        return self.U @ self.V

    @torch.no_grad()
    def update_rank(self, new_rank: int):
        new_rank = int(new_rank)
        new_rank = max(1, min(new_rank, min(self.in_dim, self.out_dim)))
        if new_rank == self.rank:
            return

        W = self.reconstruct_full_weight().detach()
        U_new, V_new = randomized_factorization(W, rank=new_rank)

        self.U = nn.Parameter(U_new)
        self.V = nn.Parameter(V_new)
        self.rank = new_rank
        print(f"Adjusted rank → {self.rank} for layer ({self.out_dim} x {self.in_dim})")


# ============================================================
# SYMPATHETIC GRADIENT ANALYSIS
# ============================================================

@torch.no_grad()
def compute_gradient_groups(weight_grad: torch.Tensor, sim_threshold: float = 0.9) -> int:
    """
    Given approximate gradient for full weight [out_dim, in_dim],
    cluster rows by cosine similarity and return size of largest group.
    """
    out_dim, in_dim = weight_grad.shape

    norms = weight_grad.norm(dim=1, keepdim=True) + 1e-8
    g = weight_grad / norms  # [out_dim, in_dim]

    sim = g @ g.t()  # [out_dim, out_dim]
    visited = torch.zeros(out_dim, dtype=torch.bool, device=weight_grad.device)
    largest = 1

    for i in range(out_dim):
        if visited[i]:
            continue
        mask = sim[i] >= sim_threshold
        group_indices = torch.where(mask)[0]
        visited[group_indices] = True
        largest = max(largest, group_indices.numel())

    return largest


@torch.no_grad()
def adaptive_rank_update(
    layer: AdaptiveLowRankLayer,
    epoch_idx: int,
    warmup_epochs: int,
    sim_threshold: float,
    min_rank: int,
    max_rank: int,
    smooth_alpha: float = 0.8,
    ramp_factor: float = 0.5,
):
    """
    Update layer.rank based on gradient structure, but:
    - skip adaptation during warmup epochs
    - smooth target rank with EMA
    - ramp changes gradually instead of instant jumps
    """
    if layer.U.grad is None or layer.V.grad is None:
        return

    if epoch_idx < warmup_epochs:
        return

    # Approximate grad of W = U @ V
    weight_grad = layer.U.grad @ layer.V + layer.U @ layer.V.grad

    suggested = compute_gradient_groups(weight_grad, sim_threshold)
    suggested = max(min_rank, min(max_rank, suggested))

    current = float(layer.rank)
    smoothed = smooth_alpha * current + (1.0 - smooth_alpha) * float(suggested)
    new_rank_float = current + ramp_factor * (smoothed - current)
    new_rank = int(round(max(min_rank, min(max_rank, new_rank_float))))

    layer.update_rank(new_rank)


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(embed_dim).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / embed_dim)
        angles = pos * angle_rates
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# TRANSFORMER BLOCK (LOW-RANK VERSION)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, init_rank=64):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate low-rank projections for Q, K, V and output
        self.q_proj = AdaptiveLowRankLayer(embed_dim, embed_dim, init_rank)
        self.k_proj = AdaptiveLowRankLayer(embed_dim, embed_dim, init_rank)
        self.v_proj = AdaptiveLowRankLayer(embed_dim, embed_dim, init_rank)
        self.out_proj = AdaptiveLowRankLayer(embed_dim, embed_dim, init_rank)

    def forward(self, x):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split heads
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, T, H, T]
        att = F.softmax(att, dim=-1)

        out = att @ v  # [B, T, H, D]
        out = out.view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, init_rank=64):
        super().__init__()
        self.fc1 = AdaptiveLowRankLayer(embed_dim, hidden_dim, init_rank)
        self.fc2 = AdaptiveLowRankLayer(hidden_dim, embed_dim, init_rank)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden, init_rank=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, init_rank)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden, init_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================
# FULL TRANSFORMER MODEL
# ============================================================

class TinyTransformerLowRank(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        ff_hidden=256,
        num_layers=3,
        context=64,
        init_rank=64,
    ):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=context)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden, init_rank=init_rank)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        # Final vocab head stays dense (no low-rank compression there yet)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)         # [B, T, C]
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits


# ============================================================
# TRAINING UTILS
# ============================================================

def get_batch(split, batch_size, context):
    data_split = train_data if split == "train" else val_data
    if len(data_split) <= context + 1:
        raise ValueError("Context too long for dataset length.")
    ix = torch.randint(0, len(data_split) - context - 1, (batch_size,))
    x = torch.stack([data_split[i:i+context] for i in ix])
    y = torch.stack([data_split[i+1:i+context+1] for i in ix])
    return x, y


def collect_lowrank_layers(model):
    return [m for m in model.modules() if isinstance(m, AdaptiveLowRankLayer)]


def train_lowrank(
    epochs=10,
    batch_size=16,
    context=64,
    init_rank=64,
):
    device = get_device()
    model = TinyTransformerLowRank(
        vocab_size,
        embed_dim=128,
        num_heads=4,
        ff_hidden=256,
        num_layers=3,
        context=context,
        init_rank=init_rank,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    lowrank_layers = collect_lowrank_layers(model)
    print(f"Total low-rank layers: {len(lowrank_layers)}")
    print("Initial ranks:", sorted({layer.rank for layer in lowrank_layers}))

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        losses = 0.0

        for batch_idx in range(200):  # 200 minibatches per epoch
            x, y = get_batch("train", batch_size, context)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            opt.zero_grad()
            loss.backward()

            # Adapt ranks once per epoch using first batch's gradients
            if batch_idx == 0:
                for layer in lowrank_layers:
                    # Slightly conservative warmup; same hyperparams for all layers for now
                    adaptive_rank_update(
                        layer,
                        epoch_idx=epoch,
                        warmup_epochs=3,
                        sim_threshold=0.9,
                        min_rank=8,
                        max_rank=min(layer.in_dim, layer.out_dim),
                        smooth_alpha=0.8,
                        ramp_factor=0.5,
                    )

            opt.step()
            losses += loss.item()

        avg_loss = losses / 200

        # Evaluate
        model.eval()
        x, y = get_batch("val", batch_size, context)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            val_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        dt = time.time() - t0
        avg_rank = sum(layer.rank for layer in lowrank_layers) / len(lowrank_layers)
        min_rank = min(layer.rank for layer in lowrank_layers)
        max_rank = max(layer.rank for layer in lowrank_layers)

        print(
            f"Epoch {epoch+1:02d} | {dt:4.2f}s | "
            f"Train loss {avg_loss:.4f} | Val loss {val_loss:.4f} | "
            f"Ranks avg={avg_rank:.1f}, min={min_rank}, max={max_rank}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--context", type=int, default=64)
    args = parser.parse_args()

    train_lowrank(
        epochs=args.epochs,
        batch_size=16,
        context=args.context,
        init_rank=args.init_rank,
    )
