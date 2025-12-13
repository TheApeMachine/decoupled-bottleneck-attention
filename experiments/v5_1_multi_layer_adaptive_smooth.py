import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# DEVICE SELECTION (MPS ON MAC)
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
    # MPS workaround: QR decomposition not fully implemented on MPS
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
# ADAPTIVE LOW-RANK LAYER
# ============================================================

class AdaptiveLowRankLayer(nn.Module):
    """
    Represents a weight matrix W (out_dim x in_dim) as U @ V
    with dynamic rank control.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, in_dim) * 0.02)

    def forward(self, x):
        # x: [batch, in_dim]
        return x @ self.V.t() @ self.U.t()

    def reconstruct_full_weight(self):
        return self.U @ self.V

    @torch.no_grad()
    def update_rank(self, new_rank: int):
        """
        Refactor W into the new rank using randomized factorization.
        """
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


# ============================================================
# MLP WITH MULTI-LAYER ADAPTIVE LOW-RANK
# ============================================================

class MLP_MultiAdaptive(nn.Module):
    def __init__(self, init_rank1=64, init_rank2=64):
        super().__init__()
        # 28*28 -> 512 as low-rank
        self.fc1 = AdaptiveLowRankLayer(28 * 28, 512, init_rank1)
        # 512 -> 512 as low-rank
        self.fc2 = AdaptiveLowRankLayer(512, 512, init_rank2)
        # 512 -> 10 stays dense
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# RANK ADAPTATION LOGIC (SMOOTHED + WARMUP)
# ============================================================

@torch.no_grad()
def adaptive_rank_update(
    layer: AdaptiveLowRankLayer,
    epoch_idx: int,
    warmup_epochs: int,
    sim_threshold: float,
    min_rank: int,
    max_rank: int,
    smooth_alpha: float = 0.8,   # EMA smoothing factor
    ramp_factor: float = 0.5,    # how fast to move toward target
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
        # No compression yet: allow features to form.
        return

    # Approximate grad of W = U @ V
    weight_grad = layer.U.grad @ layer.V + layer.U @ layer.V.grad

    suggested = compute_gradient_groups(weight_grad, sim_threshold)

    # Clamp raw suggestion into allowed band
    suggested = max(min_rank, min(max_rank, suggested))

    # Smooth with EMA toward suggested rank
    current = float(layer.rank)
    smoothed = smooth_alpha * current + (1.0 - smooth_alpha) * float(suggested)

    # Ramp: move only partway toward smoothed rank
    new_rank_float = current + ramp_factor * (smoothed - current)
    new_rank = int(round(max(min_rank, min(max_rank, new_rank_float))))

    layer.update_rank(new_rank)


def train_one_epoch(
    model,
    device,
    dataloader,
    optimizer,
    criterion,
    epoch_idx: int,
    total_epochs: int,
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    # Schedules / hyperparams for adaptation
    # fc1: more conservative, delayed
    fc1_warmup = 5        # epochs before we touch fc1
    fc1_min_rank = 16
    fc1_max_rank = 512
    fc1_sim_threshold = 0.92

    # fc2: can adapt earlier, more aggressively
    fc2_warmup = 1
    fc2_min_rank = 8
    fc2_max_rank = 512
    fc2_sim_threshold = 0.9

    # Optional: could vary ramp_factor over time, but keep constant for now
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Once per epoch, based on first batch's gradients, adapt ranks.
        if batch_idx == 0:
            adaptive_rank_update(
                model.fc1,
                epoch_idx=epoch_idx,
                warmup_epochs=fc1_warmup,
                sim_threshold=fc1_sim_threshold,
                min_rank=fc1_min_rank,
                max_rank=fc1_max_rank,
                smooth_alpha=0.85,
                ramp_factor=0.4,
            )
            adaptive_rank_update(
                model.fc2,
                epoch_idx=epoch_idx,
                warmup_epochs=fc2_warmup,
                sim_threshold=fc2_sim_threshold,
                min_rank=fc2_min_rank,
                max_rank=fc2_max_rank,
                smooth_alpha=0.8,
                ramp_factor=0.5,
            )

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()

    return running_loss / total, 100.0 * correct / total


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--init-rank1", type=int, default=64, help="Initial rank for fc1 (784->512).")
    parser.add_argument("--init-rank2", type=int, default=64, help="Initial rank for fc2 (512->512).")
    args = parser.parse_args()

    device = get_device()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Model
    model = MLP_MultiAdaptive(init_rank1=args.init_rank1, init_rank2=args.init_rank2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Initial ranks: fc1={args.init_rank1}, fc2={args.init_rank2}")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            device,
            train_loader,
            opt,
            criterion,
            epoch_idx=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1:02d} | {dt:5.2f}s | "
            f"Train {train_acc:5.2f}% | Val {val_acc:5.2f}% | "
            f"Ranks fc1={model.fc1.rank}, fc2={model.fc2.rank}"
        )


if __name__ == "__main__":
    main()
