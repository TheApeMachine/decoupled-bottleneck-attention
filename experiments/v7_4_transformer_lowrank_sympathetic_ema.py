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
# 1. RANK CONTROLLER (EMA + Momentum) - From v7.2
# ============================================================

class RankController:
    def __init__(
        self,
        init_rank,
        ema_decay=0.9,         # EMA smoothing
        adjust_rate=0.1,       # How fast we move toward EMA target (increased slightly)
        update_every=1,        # Check every epoch
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

        # Apply step
        self.rank += self.adjust_rate * self.velocity

        # Clamp
        self.rank = int(max(self.min_rank, min(self.rank, self.max_rank)))
        
        # Ensure rank doesn't get stuck in float limbo if velocity is small
        if abs(self.rank - self.ema_target) < 1.0 and self.rank != int(round(self.ema_target)):
             self.rank = int(round(self.ema_target))

        return self.rank

# ============================================================
# 2. SYMPATHETIC GRADIENT ANALYSIS - From v7.1
# ============================================================

@torch.no_grad()
def compute_gradient_groups(weight_grad: torch.Tensor, sim_threshold: float = 0.8) -> int:
    """
    Given approximate gradient for full weight [out_dim, in_dim],
    cluster rows by cosine similarity and return size of largest group? 
    Actually, to determine Rank, we often want the *number of groups* (diversity),
    or the *effective rank* of the gradient matrix.
    
    v7.1 implementation returned 'largest group size' which might be inverse of what we want?
    Wait, let's re-read v7.1 carefully.
    
    v7.1 logic:
    largest = max(largest, group_indices.numel())
    suggested = largest
    
    If 'largest group' is big, that means many rows are similar -> Low Rank.
    If 'largest group' is small (1), that means all rows are different -> High Rank.
    
    Wait, if the result is 'largest', and we use that as 'rank', that implies:
    High Similarity -> High 'largest' -> High Rank setting? 
    
    Actually, if many neurons are doing the same thing (high similarity), we should be able to COMPRESS them (Low Rank).
    If neurons are doing different things (low similarity), we need High Rank.
    
    Let's check v7.1 usage:
    suggested = compute_gradient_groups(...)
    suggested = max(min_rank, min(max_rank, suggested))
    
    If it returns the size of the largest cluster of similar gradients:
    - 100 rows. All identical. Largest cluster = 100. Suggested rank = 100.
    - 100 rows. All orthogonal. Largest cluster = 1. Suggested rank = 1.
    
    This seems backwards for compression. 
    If they are identical, rank 1 is sufficient.
    If they are orthogonal, rank 100 is needed.
    
    Let's switch to a standard 'Effective Rank' calculation on the gradient, 
    or just count the Number of Clusters.
    
    Let's implement 'Number of Clusters' as the proxy for Rank.
    """
    out_dim, in_dim = weight_grad.shape
    
    # Normalize rows
    norms = weight_grad.norm(dim=1, keepdim=True) + 1e-8
    g = weight_grad / norms  # [out_dim, in_dim]

    # Cosine similarity matrix
    sim = g @ g.t()  # [out_dim, out_dim]
    
    visited = torch.zeros(out_dim, dtype=torch.bool, device=weight_grad.device)
    n_clusters = 0

    for i in range(out_dim):
        if visited[i]:
            continue
        
        # Find all rows similar to row i
        mask = sim[i] >= sim_threshold
        
        # Mark them as visited
        visited[mask] = True
        n_clusters += 1

    return n_clusters

# ============================================================
# 3. HELPER: RANDOMIZED FACTORIZATION (MPS SAFE)
# ============================================================

@torch.no_grad()
def randomized_svd_resize(U, V, new_rank):
    """
    Resize W = U@V to new_rank using randomized SVD.
    """
    device = U.device
    W = U @ V # [out, in]
    m, n = W.shape
    
    # If standard SVD is available and cheap, use it (on CPU if MPS issues)
    # But randomized is often better for 'approximate' low rank
    
    # Let's do a simple SVD on CPU to be safe and accurate for these small sizes
    W_cpu = W.detach().cpu()
    
    try:
        # full_matrices=False is generally faster
        Uw, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        
        # Truncate
        Uw = Uw[:, :new_rank]
        S = S[:new_rank]
        Vh = Vh[:new_rank, :]
        
        # Distribute S (sqrt) to keep U and V balanced
        S_sqrt = torch.diag(torch.sqrt(S))
        
        U_new = Uw @ S_sqrt
        V_new = S_sqrt @ Vh
        
        return U_new.to(device), V_new.to(device)
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Fallback: Random init (worst case)
        return (torch.randn(m, new_rank, device=device) * 0.02, 
                torch.randn(new_rank, n, device=device) * 0.02)


# ============================================================
# 4. LOW RANK LINEAR LAYER (COMBINED)
# ============================================================

class SympatheticLowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Controller
        self.rank_controller = RankController(
            rank, 
            ema_decay=0.9, 
            adjust_rate=0.2,
            min_rank=8,
            max_rank=min(in_features, out_features)
        )

        # Initial factors
        self.U = nn.Parameter(torch.randn(out_features, rank) / math.sqrt(out_features))
        self.V = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))

    def forward(self, x):
        # x: [..., in]
        # U: [out, r], V: [r, in]
        # W = U V.  y = x W^T = x (V^T U^T)
        return x @ self.V.t() @ self.U.t()

    def get_suggested_rank(self):
        # Reconstruct effective Gradient of W
        # dL/dW approx = dL/dU * V^T + U^T * dL/dV (dimensions need care)
        # U: [O, R], V: [R, I]
        # U.grad: [O, R], V.grad: [R, I]
        # W = U @ V => [O, I]
        
        if self.U.grad is None or self.V.grad is None:
            return self.rank_controller.rank

        with torch.no_grad():
            # Grad(W) contribution from U branch: U.grad @ V
            # Grad(W) contribution from V branch: U @ V.grad
            # This is the product rule for matrix multiplication
            
            G_W = self.U.grad @ self.V + self.U @ self.V.grad
            
            # Analyze diversity of this gradient matrix
            # If gradients are diverse -> we need more rank to capture them
            # If gradients are similar -> we can compress
            n_clusters = compute_gradient_groups(G_W, sim_threshold=0.85)
            
            return n_clusters

    def apply_rank_update(self):
        suggested = self.get_suggested_rank()
        new_rank = self.rank_controller.update(suggested)
        
        current_rank = self.U.shape[1]
        
        if new_rank == current_rank:
            return

        # Resize
        new_U, new_V = randomized_svd_resize(self.U.data, self.V.data, new_rank)
        self.U = nn.Parameter(new_U)
        self.V = nn.Parameter(new_V)
        
        # Important: Reset optimizer state for these parameters? 
        # In simple loop, Adam keeps running variance. Changing shape breaks it.
        # We usually just zero grad and let Adam rebuild moments. 
        # Ideally we'd migrate Adam state, but for "Tiny" experiments, reset is acceptable.
        # (Parameter replacement automatically gives them None grads)
            
        print(f"Layer ({self.out_features}x{self.in_features}) rank: {current_rank} -> {new_rank} (Suggested: {suggested})")


# ============================================================
# 5. TRANSFORMER BLOCK
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
        for layer in [
            self.attn_q, self.attn_k, self.attn_v, self.attn_o,
            self.ff1, self.ff2
        ]:
            layer.apply_rank_update()


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

# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train_model(epochs=20, rank=64, lr=3e-4):
    data = torch.tensor([stoi[c] for c in tiny_text], device=device)
    train = data[:-100] 
    val = data[-100:]

    model = TinyTransformer(len(chars), rank=rank).to(device)
    
    # We need to recreate optimizer if parameters change?
    # Actually, Adam holds refs to params. If we replace params in Module, 
    # we MUST update optimizer param_groups or re-init optimizer.
    # Simple strategy: Re-init optimizer, but try to carry over momentum?
    # Hard. For this experiment, let's just Re-init optimizer or use a simple SGD to see.
    # Or: "param_groups" hacking.
    
    # Strategy: Just define optimizer outside loop, and inside loop, 
    # if ranks change, we rebuild optimizer. 
    # This loses momentum history, but acceptable for dynamic rank proof-of-concept.
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    B = 4 # Small batch for tiny data
    T = 16

    def get_batch(split):
        d = train if split == "train" else val
        if len(d) <= T: # Safety
            return d[:-1].unsqueeze(0), d[1:].unsqueeze(0)
            
        ix = torch.randint(len(d)-T, (B,))
        x = torch.stack([d[i:i+T] for i in ix])
        y = torch.stack([d[i+1:i+T+1] for i in ix])
        return x, y

    print(f"Training on {device}...")
    print(f"Initial Rank: {rank}")
    
    for epoch in range(1, epochs+1):
        model.train()
        
        # 10 steps per epoch
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

        # Rank Update Step
        if epoch > 2: # Warmup
            model.update_all_ranks()
            
            # Re-bind optimizer to new parameters
            # (Because we replaced nn.Parameter objects in the layers)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init-rank", type=int, default=64)
    args = parser.parse_args()

    train_model(epochs=args.epochs, rank=args.init_rank)

