import math
import time
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------
# Device selection (Metal/MPS)
# ------------------------------

def get_device():
    if torch.backends.mps.is_available():
        print("Using Metal (MPS) backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")


# ------------------------------
# Simple MLP model
# ------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------------------
# Gradient grouping logic
# ------------------------------

def compute_groups_from_grad(
    weight_grad: torch.Tensor,
    sim_threshold: float = 0.9
) -> List[List[int]]:
    """
    Given a weight gradient tensor of shape [out_features, in_features],
    find groups of "sympathetic neurons" based on cosine similarity
    of their gradient vectors.

    Very simple greedy clustering:
    - Normalize each row of grads to unit length.
    - Compute cosine similarity matrix.
    - For each unassigned neuron i, create a group of all j
      with similarity >= sim_threshold.
    """
    # weight_grad: [out, in]
    out_features, in_features = weight_grad.shape

    # Normalize gradients per neuron to unit norm
    grads = weight_grad.detach()
    norms = grads.norm(dim=1, keepdim=True) + 1e-8
    g = grads / norms  # [out, in]

    # Cosine similarity between all pairs: [out, out]
    sim = g @ g.t()

    # Greedy grouping
    unassigned = set(range(out_features))
    groups = []

    while unassigned:
        i = unassigned.pop()
        # Neurons with high similarity to i
        mask = sim[i] >= sim_threshold
        group = [i]
        for j in list(unassigned):
            if mask[j].item():
                group.append(j)
                unassigned.remove(j)
        groups.append(group)

    return groups


def apply_grouped_gradients(
    weight_param: torch.nn.Parameter,
    groups: List[List[int]]
):
    """
    For each group of neuron indices, replace their gradients with
    the mean gradient of the group.

    weight_param: nn.Parameter of shape [out_features, in_features]
    groups: list of lists of neuron indices (rows).
    """
    if weight_param.grad is None:
        return

    grads = weight_param.grad
    out_features, in_features = grads.shape

    for group in groups:
        if len(group) <= 1:
            continue
        idx = torch.tensor(group, device=grads.device, dtype=torch.long)
        group_grad = grads[idx].mean(dim=0, keepdim=True)  # [1, in]
        grads[idx] = group_grad


# ------------------------------
# Training / evaluation helpers
# ------------------------------

def train_one_epoch(
    model,
    device,
    dataloader,
    optimizer,
    criterion,
    mode="baseline",
    sim_threshold=0.9,
    coarse_to_fine_schedule=None,
    epoch_idx=0,
    target_layer_name="fc2",
):
    """
    mode:
      - "baseline": normal training
      - "grouped": group gradients every step
      - "coarse_to_fine": use schedule to control grouping strength
    coarse_to_fine_schedule:
      - function(epoch_idx) -> sim_threshold or None
        if returns None: no grouping in that epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Decide if we do grouping this step
        do_grouping = False
        threshold = sim_threshold

        if mode == "grouped":
            do_grouping = True
        elif mode == "coarse_to_fine" and coarse_to_fine_schedule is not None:
            threshold = coarse_to_fine_schedule(epoch_idx)
            if threshold is not None:
                do_grouping = True

        if do_grouping:
            # Find the target layer
            target_layer = dict(model.named_modules())[target_layer_name]
            weight_param = target_layer.weight

            # Compute groups from gradient
            groups = compute_groups_from_grad(
                weight_param.grad, sim_threshold=threshold
            )

            # Apply grouped gradient update
            apply_grouped_gradients(weight_param, groups)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ------------------------------
# Coarse-to-fine schedule example
# ------------------------------

def example_coarse_to_fine_schedule(num_epochs):
    """
    Returns a function(epoch_idx) -> sim_threshold or None.

    Example:
    - Epochs [0 .. 1]: very coarse, strong grouping (threshold = 0.7)
    - Epochs [2 .. 3]: medium grouping (threshold = 0.85)
    - Epochs [4 .. end]: no grouping (None)
    """
    def schedule(epoch_idx):
        if epoch_idx < 2:
            return 0.7
        elif epoch_idx < 4:
            return 0.85
        else:
            return None

    return schedule


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "grouped", "coarse_to_fine"],
        help="Training mode."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size."
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for grouping (used in 'grouped' mode)."
    )
    args = parser.parse_args()

    device = get_device()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2
    )

    # Model / optimizer / loss
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Optional coarse-to-fine schedule
    ctf_schedule = None
    if args.mode == "coarse_to_fine":
        ctf_schedule = example_coarse_to_fine_schedule(args.epochs)

    print(f"Mode: {args.mode}")
    print("Training...")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            mode=args.mode,
            sim_threshold=args.sim_threshold,
            coarse_to_fine_schedule=ctf_schedule,
            epoch_idx=epoch,
            target_layer_name="fc2",
        )
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} "
            f"| time {dt:5.1f}s "
            f"| train_loss {train_loss:.4f} acc {train_acc:5.2f}% "
            f"| val_loss {val_loss:.4f} acc {val_acc:5.2f}%"
        )

    print("Done.")


if __name__ == "__main__":
    main()
