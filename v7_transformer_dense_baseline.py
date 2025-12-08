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
# TINY CHARACTER DATASET
# ============================================================

# We use a tiny text dataset (Shakespeare-ish) directly in code.
# Later we replace it with real dataset or tokenization pipeline.
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

# Build vocabulary
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
# TRANSFORMER UTILITIES
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
# TRANSFORMER BLOCK (DENSE BASELINE)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)

        # Split heads
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, T, H, T]
        att = F.softmax(att, dim=-1)

        out = att @ v  # [B, T, H, D]
        out = out.view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ============================================================
# FULL TRANSFORMER MODEL
# ============================================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_hidden=256, num_layers=3, context=64):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=context)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden)
        for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)                      # [B, T, C]
        x = self.pos(x)                          # add positional encoding
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits


# ============================================================
# TRAINING LOOP
# ============================================================

def get_batch(split, batch_size, context):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data_split) - context - 1, (batch_size,))
    x = torch.stack([data_split[i:i+context] for i in ix])
    y = torch.stack([data_split[i+1:i+context+1] for i in ix])
    return x, y

def train_baseline(epochs=5, batch_size=16, context=64):
    device = get_device()
    model = TinyTransformer(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Training dense baseline transformer...")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        losses = 0
        for _ in range(200):  # 200 minibatches per epoch
            x, y = get_batch("train", batch_size, context)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
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
        print(f"Epoch {epoch+1:02d} | {dt:4.2f}s | Train loss {avg_loss:.4f} | Val loss {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_baseline(epochs=args.epochs)
