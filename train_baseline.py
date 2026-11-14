"""
train_baseline.py
=================
Step 1 of Question 1: train the FP32 ResNet-18 baseline on CIFAR-10 and
save its checkpoint. PTQ, QAT and the ONNX export all start from this
checkpoint.

Run:  python train_baseline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

import config
from common import (ResNet18CIFAR, get_train_loader, get_test_loader,
                    evaluate, set_seed, ensure_dirs)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """
    Run one full pass over the training data.

    Returns:
        The average training loss over the epoch.
    """
    model.train()
    running_loss, batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                    # clear old gradients
        loss = criterion(model(images), labels)  # forward pass + loss
        loss.backward()                          # backpropagation
        optimizer.step()                         # update weights

        running_loss += loss.item()
        batches += 1
    return running_loss / batches


def main() -> None:
    set_seed()
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    model = ResNet18CIFAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.BASELINE_LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)
    # Cosine schedule smoothly lowers the learning rate to ~0 by the last epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.BASELINE_EPOCHS)

    for epoch in range(1, config.BASELINE_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}/{config.BASELINE_EPOCHS} | "
              f"loss {loss:.4f} | test accuracy {accuracy:.2f}%")

    # Save only the weights (state_dict), moved to CPU for portability
    torch.save(model.cpu().state_dict(), config.FP32_PATH)
    print(f"Saved FP32 checkpoint to {config.FP32_PATH}")


if __name__ == "__main__":
    main()
