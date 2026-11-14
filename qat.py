"""
qat.py
======
Step 3 of Question 1: Quantization-Aware Training (QAT).

Unlike PTQ, QAT simulates INT8 rounding DURING training, so the weights
learn to cope with the quantization error.

Pipeline:
  1. Load the pretrained FP32 model.
  2. Fuse layers (QAT version, model in train mode).
  3. Attach a QAT qconfig and call prepare_qat(): this inserts "fake
     quantization" modules that round values to INT8 levels in the forward
     pass but keep everything in float so gradients still flow.
  4. Fine-tune for a few epochs.
  5. Move to CPU, switch to eval mode and convert() to a real INT8 model.
  6. Evaluate and save.

Run:  python qat.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

import config
from common import (ResNet18CIFAR, fuse_model, get_train_loader,
                    get_test_loader, evaluate, set_seed, ensure_dirs)


def main() -> None:
    set_seed()
    ensure_dirs()
    # Fake-quantized training can use the GPU; the final INT8 model is CPU-only
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"QAT fine-tuning on: {device}")

    # Step 1: initialise from the pretrained FP32 weights
    model = ResNet18CIFAR()
    model.load_state_dict(torch.load(config.FP32_PATH, map_location="cpu"))

    # Step 2: fuse layers. QAT needs the model in TRAIN mode for fusion.
    model.train()
    fuse_model(model, qat=True)

    # Step 3: attach QAT config and insert fake-quantization modules
    torch.backends.quantized.engine = config.QUANT_BACKEND
    model.qconfig = get_default_qat_qconfig(config.QUANT_BACKEND)
    prepare_qat(model, inplace=True)
    model.to(device)

    # Step 4: fine-tune with a small learning rate
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.QAT_LR,
                          momentum=config.MOMENTUM,
                          weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, config.QAT_EPOCHS + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)  # forward with fake-quant
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        print(f"QAT epoch {epoch}/{config.QAT_EPOCHS} | "
              f"loss {running_loss / batches:.4f}")

    # Step 5: convert to a real INT8 model (must be on CPU, in eval mode)
    model.cpu().eval()
    convert(model, inplace=True)

    # Step 6: evaluate and save
    accuracy = evaluate(model, get_test_loader(), device="cpu")
    print(f"QAT INT8 test accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), config.QAT_PATH)
    print(f"Saved QAT INT8 checkpoint to {config.QAT_PATH}")


if __name__ == "__main__":
    main()
