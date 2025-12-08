import math
import time
import argparse
from typing import Optional, Callable

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
    return torch.device("cpu")


# ============================================================
# ADAPTIVE LOW-RANK LAYER
# ============================================================

class AdaptiveLowRankLayer(nn.Module):
    """
    Learnable layer that maintains a low-rank factorization W = U @ V.
    Rank r is dynamically adjusted during training.
    """
    def __init__(self, in_dim=512, out_dim=512, rank=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        # U: out_dim × rank
        # V: rank × in_dim
        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, in_dim) * 0.02)

    def forward(self, x):
        # x: [batch, in_dim]
        # return: x @ (V.T @ U.T)
        return x @ self.V.t() @ self.U.t()

    def reconstruct_full_weight(self):
        return self.U @ self.V

    @torch.no_grad()
    def update_rank(self, new_rank):
        """
        Re-factorize current weight matrix to match new rank via SVD.
        """
        if new_rank == self.rank:
            return  # no change

        W = self.reconstruct_full_weight()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Pick the top new_rank singular values
        U_new = U[:, :new_rank]
        S_new = torch.diag(S[:new_rank])
        V_new = Vt[:new_rank, :]

        # Recreate factorization: W ≈ U_new * S_new * V_new
        # So new factors (U', V') must satisfy:
        # W = (U_new * sqrt(S_new)) @ (sqrt(S_new) * V_new)
        SR = torch.sqrt(S_new)
        self.U = nn.Parameter(U_new @ SR)
        self.V = nn.Parameter(SR @ V_new)

        self.rank = new_rank
        print(f"Adjusted rank → {self.rank}")


# ============================================================
# SYMPATHETIC GRADIENT ANALYSIS
# ============================================================

@torch.no_grad()
def compute_gradient_groups(weight_grad: torch.Tensor, sim_threshold: float = 0.9):
    """
    Detect sympathetic neuron clusters based on gradient similarity.
    Returns largest cluster size.
    """
    out_dim, in_dim = weight_grad.shape

    norms = weight_grad.norm(dim=1, keepdim=True) + 1e-8
    g = weight_grad / norms

    sim = g @ g.t()  # [out, out]

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
# MODEL WITH LOW-RANK FC2
# ============================================================

class MLP_AdaptiveLowRank(nn.Module):
    def __init__(self, rank=64):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = AdaptiveLowRankLayer(512, 512, rank)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# TRAINING & EVAL
# ============================================================

def train_one_epoch(
    model, device, dataloader, optimizer, criterion,
    epoch_idx, adjust_every=1, min_rank=16, max_rank=512,
    sim_threshold=0.9,
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

        # Only adjust rank once per epoch
        if batch_idx == 0 and epoch_idx % adjust_every == 0:
            fc2 = model.fc2
            weight_grad = fc2.reconstruct_full_weight().grad \
                          if fc2.reconstruct_full_weight().grad is not None \
                          else None

            if fc2.U.grad is not None:
                # approximate gradient for full weight by reconstructing gradient W' = U.grad @ V + U @ V.grad
                weight_grad = (
                    fc2.U.grad @ fc2.V +
                    fc2.U @ fc2.V.grad
                )

                largest = compute_gradient_groups(weight_grad, sim_threshold)

                new_rank = max(min_rank, min(max_rank, largest))
                fc2.update_rank(new_rank)

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
    test_loader = DataLoader(test_ds, batch_size=256)

    # Model
    model = MLP_AdaptiveLowRank(rank=args.init_rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Initial rank = {args.init_rank}")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, opt, criterion,
            epoch_idx=epoch,
            adjust_every=1,
            min_rank=16,
            max_rank=512,
            sim_threshold=0.9,
        )
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch+1} | {dt:.2f}s | Train {train_acc:5.2f}% | Val {val_acc:5.2f}% | Rank {model.fc2.rank}")


if __name__ == "__main__":
    main()
