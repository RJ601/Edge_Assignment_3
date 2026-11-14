"""
prune_utils.py
==============
Helper functions shared by the two pruning scripts and the pruning benchmark
(Question 2):

  1. count_parameters        total and non-zero parameter counts
  2. prunable_sparsity       % of zero weights in all conv + fully connected layers
  3. layer_sparsity_rows     per-layer sparsity table (for the Excel report)
  4. channel_rows            per-block channel counts before/after structured pruning
  5. fine_tune               short retraining to recover accuracy after pruning
"""

import torch
import torch.nn as nn
import torch.optim as optim

import config


def count_parameters(model: nn.Module):
    """
    Count the parameters of `model`.

    Returns:
        (total, nonzero): all learnable numbers, and how many of them are not
        exactly zero. For an unstructured-pruned model `nonzero` is much
        smaller than `total`. For a structured-pruned model `total` itself
        is smaller, because whole channels were physically removed.
    """
    total, nonzero = 0, 0
    for parameter in model.parameters():
        total += parameter.numel()
        nonzero += int((parameter != 0).sum().item())
    return total, nonzero


def prunable_layers(model: nn.Module):
    """Return (name, module) for every convolution and fully connected layer."""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]


def prunable_sparsity(model: nn.Module) -> float:
    """
    Percentage of weights that are exactly zero across all conv + fc layers.

    Works both while masks are active (module.weight is then weight*mask)
    and after pruning has been made permanent.
    """
    zeros, total = 0, 0
    for _, module in prunable_layers(model):
        zeros += int((module.weight == 0).sum().item())
        total += module.weight.numel()
    return 100.0 * zeros / total


def layer_sparsity_rows(model: nn.Module):
    """One dict per conv/fc layer: its name, weight count, zeros and sparsity %."""
    rows = []
    for name, module in prunable_layers(model):
        zeros = int((module.weight == 0).sum().item())
        total = module.weight.numel()
        rows.append({
            "layer": name,
            "weights": total,
            "zero_weights": zeros,
            "sparsity_percent": 100.0 * zeros / total,
        })
    return rows


def channel_rows(dense_model: nn.Module, pruned_model: nn.Module):
    """
    One dict per residual block comparing its inner channel count in the
    original (dense) model with the structured-pruned model.
    """
    # Local import avoids a circular import at module-load time
    from common import BasicBlock

    dense_blocks = [(n, m) for n, m in dense_model.named_modules()
                    if isinstance(m, BasicBlock)]
    pruned_blocks = [m for _, m in pruned_model.named_modules()
                     if isinstance(m, BasicBlock)]

    rows = []
    for (name, dense_block), pruned_block in zip(dense_blocks, pruned_blocks):
        before = dense_block.conv1.out_channels
        after = pruned_block.conv1.out_channels
        rows.append({
            "block": name,
            "channels_before": before,
            "channels_after": after,
            "pruned_percent": 100.0 * (before - after) / before,
        })
    return rows


def fine_tune(model: nn.Module, loader, epochs: int, lr: float,
              device: str, label: str = "") -> None:
    """
    Retrain `model` for a few epochs so it recovers the accuracy lost to pruning.

    A fresh SGD optimiser is created on every call, and the learning rate
    decays smoothly (cosine) from `lr` towards zero over the given epochs.
    Masks created by torch pruning stay active during fine-tuning, so pruned
    weights remain zero.

    Args:
        model:  the model to train (already on `device`).
        loader: the augmented CIFAR-10 training loader.
        epochs: how many passes over the training data.
        lr:     starting learning rate.
        device: "cuda" or "cpu".
        label:  text shown in the progress lines, e.g. "iteration 2".
    """
    if epochs <= 0:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        scheduler.step()
        print(f"  fine-tune {label} | epoch {epoch}/{epochs} | "
              f"loss {running_loss / batches:.4f}")
