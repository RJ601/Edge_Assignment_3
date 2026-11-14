"""
ptq.py
======
Step 2 of Question 1: Post-Training Static Quantization (PTQ).

Pipeline:
  1. Load the trained FP32 model.
  2. Fuse Conv+BN(+ReLU) layers.
  3. Choose the quantization backend and attach a quantization config.
  4. prepare():  insert observers that record activation ranges.
  5. Calibrate:  run a subset of training data through the model so the
                 observers learn the typical value range at each layer.
  6. convert():  replace FP32 layers with INT8 layers using those ranges.
  7. Evaluate and save the INT8 model.

No training happens here: the weights are only rescaled/rounded to INT8.

Run:  python ptq.py
"""

import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert

import config
from common import (ResNet18CIFAR, fuse_model, get_calibration_loader,
                    get_test_loader, evaluate, set_seed, ensure_dirs)


def calibrate(model, loader) -> None:
    """
    Feed calibration data through the prepared model (forward passes only).

    The observers inserted by prepare() watch the activations flowing
    through and record their min/max, which later become the INT8 scales.
    """
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:   # labels are not needed for calibration
            model(images)


def main() -> None:
    set_seed()
    ensure_dirs()

    # Step 1: load the trained FP32 model (quantization runs on CPU)
    model = ResNet18CIFAR()
    model.load_state_dict(torch.load(config.FP32_PATH, map_location="cpu"))
    model.eval()

    # Step 2: fuse compatible layers (must be done in eval mode for PTQ)
    fuse_model(model, qat=False)

    # Step 3: configure the backend and the quantization settings
    torch.backends.quantized.engine = config.QUANT_BACKEND
    model.qconfig = get_default_qconfig(config.QUANT_BACKEND)
    print(f"Backend: {config.QUANT_BACKEND}")

    # Step 4: insert observers
    prepare(model, inplace=True)

    # Step 5: calibrate using a subset of the training data
    calibration_loader = get_calibration_loader()
    print(f"Calibrating with {config.CALIBRATION_SAMPLES} training images...")
    calibrate(model, calibration_loader)

    # Step 6: convert to a real INT8 model
    convert(model, inplace=True)

    # Step 7: evaluate and save
    accuracy = evaluate(model, get_test_loader(), device="cpu")
    print(f"PTQ INT8 test accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), config.PTQ_PATH)
    print(f"Saved PTQ INT8 checkpoint to {config.PTQ_PATH}")


if __name__ == "__main__":
    main()
