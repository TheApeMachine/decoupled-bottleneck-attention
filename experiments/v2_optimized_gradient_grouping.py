import math
import time
import argparse
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------
# Device selection
# ------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------------------
# Optimized Gradient Grouping
# ------------------------------

@torch.no_grad()
def compute_neuron_labels(
    weight_grad: torch.Tensor,
    sim_threshold: float = 0.9
) -> torch.Tensor:
    """
    Returns a tensor of shape [out_features] containing the group ID for each neuron.
    
    Optimization:
    - Sorts by gradient norm so 'strong' neurons act as cluster centroids.
    - Uses boolean masking on GPU to avoid Python loops over indices.
    """
    out_features, _ = weight_grad.shape
    
    # 1. Normalize gradients
    norms = weight_grad.norm(dim=1, keepdim=True) + 1e-8
    g = weight_grad / norms  # [out, in]

    # 2. Compute Similarity Matrix [out, out]
    sim_matrix = g @ g.t()

    # 3. Prioritized Greedy Clustering
    # We process neurons with highest norms first to serve as centroids
    sorted_indices = torch.argsort(norms.squeeze(), descending=True)
    
    labels = torch.full((out_features,), -1, device=weight_grad.device, dtype=torch.long)
    current_group_id = 0
    
    # We still need a loop, but we operate on masks. 
    # Since we mark 'assigned' neurons, the loop shrinks effectively.
    # Note: For massive layers, this loop is still the bottleneck, 
    # which is why 'grouping_interval' is crucial.
    
    for i in sorted_indices:
        if labels[i] != -1:
            continue
            
        # Find sympathetic neurons (sim > threshold) that are not yet assigned
        # sim_matrix[i] shape is [out]
        sympathetic_mask = (sim_matrix[i] >= sim_threshold)
        
        # Only take those that haven't been assigned yet
        # (Logic: If A is sim to B, and B is already in Group X, 
        # strictly greedy means A starts a new group or joins B. 
        # Here we start a new group to keep centroids distinct.)
        unassigned_mask = (labels == -1)
        final_mask = sympathetic_mask & unassigned_mask
        
        # Assign group ID
        labels[final_mask] = current_group_id
        
        # Ensure the centroid itself is assigned (in case sim[i,i] < threshold due to float error)
        labels[i] = current_group_id
        
        current_group_id += 1

    return labels


@torch.no_grad()
def apply_grouped_gradients_vectorized(
    weight_param: torch.nn.Parameter,
    labels: torch.Tensor
):
    """
    Replaces gradients with group averages using vectorized scatter/gather operations.
    Avoids Python loops entirely.
    """
    if weight_param.grad is None:
        return

    grads = weight_param.grad
    num_groups = labels.max().item() + 1
    
    # 1. Sum gradients per group
    # shape: [num_groups, in_features]
    group_sums = torch.zeros((num_groups, grads.shape[1]), device=grads.device, dtype=grads.dtype)
    group_sums.index_add_(0, labels, grads)
    
    # 2. Count neurons per group
    # shape: [num_groups, 1]
    ones = torch.ones((grads.shape[0], 1), device=grads.device, dtype=grads.dtype)
    group_counts = torch.zeros((num_groups, 1), device=grads.device, dtype=grads.dtype)
    group_counts.index_add_(0, labels, ones)
    
    # 3. Compute means
    group_means = group_sums / (group_counts + 1e-8)
    
    # 4. Broadcast back to original neurons
    # labels shape: [out_features] -> expand to [out_features, in_features] for gather?
    # Actually, simple indexing works:
    grads.copy_(group_means[labels])


# ------------------------------
# Training Logic
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
    grouping_interval=10  # Optimization: Don't group every step
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Cache the target layer to avoid lookup every iteration
    target_layer = None
    if mode != "baseline":
        target_layer = dict(model.named_modules())[target_layer_name]

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # --- Grouping Logic ---
        # Only run if mode is active AND we hit the interval
        if mode != "baseline" and (batch_idx % grouping_interval == 0):
            
            # Determine threshold
            threshold = sim_threshold
            if mode == "coarse_to_fine" and coarse_to_fine_schedule:
                threshold = coarse_to_fine_schedule(epoch_idx)

            if threshold is not None:
                # 1. Compute Labels (The expensive part)
                labels = compute_neuron_labels(
                    target_layer.weight.grad, 
                    sim_threshold=threshold
                )
                if batch_idx == 0:  # just log once per epoch
                    num_groups = labels.max().item() + 1
                    # group sizes
                    sizes = torch.bincount(labels, minlength=num_groups)
                    print(f"Epoch {epoch_idx} | groups: {num_groups} | "
                        f"min: {sizes.min().item()} max: {sizes.max().item()} "
                        f"mean: {sizes.float().mean().item():.2f}")

                # 2. Apply Averaging (The fast part)
                apply_grouped_gradients_vectorized(target_layer.weight, labels)

        optimizer.step()

        # Stats
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


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

    return running_loss / total, 100.0 * correct / total


def example_coarse_to_fine_schedule(num_epochs):
    def schedule(epoch_idx):
        # First 30%: Strong grouping
        if epoch_idx < num_epochs * 0.3:
            return 0.75
        # Middle 40%: Moderate grouping
        elif epoch_idx < num_epochs * 0.7:
            return 0.90
        # Last 30%: No grouping (fine tuning)
        else:
            return None
    return schedule


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="coarse_to_fine", choices=["baseline", "grouped", "coarse_to_fine"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sim-threshold", type=float, default=0.9)
    parser.add_argument("--grouping-interval", type=int, default=10, 
                        help="Apply grouping every N batches to save time.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Data - Added pin_memory for speed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ctf_schedule = None
    if args.mode == "coarse_to_fine":
        ctf_schedule = example_coarse_to_fine_schedule(args.epochs)

    print(f"Mode: {args.mode} | Interval: {args.grouping_interval}")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion,
            mode=args.mode,
            sim_threshold=args.sim_threshold,
            coarse_to_fine_schedule=ctf_schedule,
            epoch_idx=epoch,
            target_layer_name="fc2",
            grouping_interval=args.grouping_interval
        )
        val_loss, val_acc = evaluate(model, device, test_loader, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch+1:02d} | {dt:5.1f}s | Train: {train_loss:.4f} ({train_acc:5.2f}%) | Val: {val_loss:.4f} ({val_acc:5.2f}%)")

if __name__ == "__main__":
    main()