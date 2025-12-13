import time
import argparse
from typing import Optional

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
# RANDOMIZED LOW-RANK FACTORIZATION (GPU-FRIENDLY)
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
    using randomized SVD-like factorization.

    - Heavy matmuls + QR happen on the current device (MPS/GPU).
    - Only a small (k x n) matrix is sent to CPU for SVD, where
      k = rank + oversample is small.
    """
    device = W.device
    m, n = W.shape
    k = min(rank + oversample, min(m, n))

    # Random projection
    Omega = torch.randn(n, k, device=device)
    Y = W @ Omega  # [m, k]

    # Power iterations to sharpen spectrum a bit
    for _ in range(n_iter):
        Y = W @ (W.t() @ Y)

    # Orthonormal basis for the subspace
    # MPS workaround: QR decomposition not fully implemented on MPS
    if Y.device.type == "mps":
        Q_cpu, _ = torch.linalg.qr(Y.cpu(), mode="reduced")
        Q = Q_cpu.to(Y.device)
    else:
        Q, _ = torch.linalg.qr(Y, mode="reduced")  # [m, k]

    # Small matrix
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
    # W ≈ (Q @ U_hat) @ (diag(S) @ Vt)
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
    W is represented as U @ V with dynamic rank.
    """
    def __init__(self, in_dim=512, out_dim=512, rank=64):
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
        Use randomized factorization to refactor W into new rank.
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
        print(f"Adjusted rank → {self.rank}")


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
# MODEL WITH ADAPTIVE LOW-RANK FC2
# ============================================================

class MLP_AdaptiveLowRank(nn.Module):
    def __init__(self, init_rank=64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = AdaptiveLowRankLayer(512, 512, init_rank)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_one_epoch(
    model,
    device,
    dataloader,
    optimizer,
    criterion,
    epoch_idx: int,
    adjust_every: int = 1,
    min_rank: int = 8,
    max_rank: int = 512,
    sim_threshold: float = 0.9,
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Rank adjustment once per epoch (using gradients from the first batch)
        if batch_idx == 0 and (epoch_idx % adjust_every == 0):
            fc2 = model.fc2
            if fc2.U.grad is not None and fc2.V.grad is not None:
                # Approximate grad of full W = U @ V
                # d(UV)/dθ ≈ U.grad @ V + U @ V.grad
                weight_grad = fc2.U.grad @ fc2.V + fc2.U @ fc2.V.grad

                largest = compute_gradient_groups(weight_grad, sim_threshold)
                new_rank = max(min_rank, min(max_rank, largest))
                fc2.update_rank(new_rank)

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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--init-rank", type=int, default=64)
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
    model = MLP_AdaptiveLowRank(init_rank=args.init_rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Initial rank = {args.init-rank if hasattr(args, 'init-rank') else args.init_rank}")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            device,
            train_loader,
            opt,
            criterion,
            epoch_idx=epoch,
            adjust_every=1,
            min_rank=8,
            max_rank=512,
            sim_threshold=0.9,
        )
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1} | {dt:5.2f}s | "
            f"Train {train_acc:5.2f}% | Val {val_acc:5.2f}% | "
            f"Rank {model.fc2.rank}"
        )


if __name__ == "__main__":
    main()
