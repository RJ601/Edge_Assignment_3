"""
export_onnx.py
==============
Step 4 of Question 1: export the trained FP32 PyTorch model to ONNX.

ONNX is a framework-neutral model format. Once exported, the model can be
run with ONNX Runtime on any CPU without needing PyTorch installed.

Pipeline:
  1. Load the trained FP32 checkpoint.
  2. Export it to an .onnx file. The batch dimension is marked "dynamic"
     so the same file works for 1 image or for 128 images.
  3. Check that the .onnx file is structurally valid.
  4. Sanity-check: PyTorch and ONNX Runtime must give (almost) the same
     outputs for the same input.

The two helper functions below (export_model, verify_export) work for ANY
model kind, so export_onnx_pruned.py (Question 2) reuses them.

Run:  python export_onnx.py
"""

import numpy as np
import onnx
import torch

import config
from common import (load_model, ONNXModel, MODEL_REGISTRY, set_seed,
                    ensure_dirs, require_file)


def export_model(model: torch.nn.Module, onnx_path: str) -> None:
    """Export an eval-mode CPU PyTorch `model` to the file `onnx_path`."""
    # Export works by running one example input through the model and
    # recording every operation. Only the SHAPE of the example matters.
    example_input = torch.randn(1, 3, 32, 32)

    export_args = dict(
        input_names=[config.ONNX_INPUT_NAME],
        output_names=[config.ONNX_OUTPUT_NAME],
        opset_version=config.ONNX_OPSET,
        # Axis 0 (the batch size) may change at run time
        dynamic_axes={
            config.ONNX_INPUT_NAME: {0: "batch"},
            config.ONNX_OUTPUT_NAME: {0: "batch"},
        },
        do_constant_folding=True,  # pre-compute fixed parts (also folds BN into conv)
    )

    try:
        # Newer PyTorch versions: dynamo=False selects the classic, stable exporter
        torch.onnx.export(model, example_input, onnx_path,
                          dynamo=False, **export_args)
    except TypeError:
        # Older PyTorch versions do not have the "dynamo" argument
        torch.onnx.export(model, example_input, onnx_path, **export_args)
    print(f"Exported ONNX model to {onnx_path}")


def verify_export(kind: str, onnx_path: str) -> None:
    """
    Validate the ONNX file and compare its outputs with the PyTorch model.

    Args:
        kind:      the PyTorch model kind ("fp32", "unstructured", "structured").
        onnx_path: the exported file to check.
    """
    # Structural check (raises an error if the file is malformed)
    onnx.checker.check_model(onnx.load(onnx_path))
    print("ONNX structure check: passed")

    # Numerical check on a random batch of 8 images
    batch = torch.randn(8, 3, 32, 32)
    with torch.inference_mode():
        torch_logits = load_model(kind)(batch).numpy()
        onnx_logits = ONNXModel(onnx_path)(batch).numpy()

    # Compare relative to the size of the outputs, so the check is fair
    # whether the logits are around 10 or around 10,000
    max_difference = float(np.abs(torch_logits - onnx_logits).max())
    scale = max(1.0, float(np.abs(torch_logits).max()))
    relative_difference = max_difference / scale
    same_predictions = bool((torch_logits.argmax(1) == onnx_logits.argmax(1)).all())

    print(f"Largest PyTorch-vs-ONNX output difference: {max_difference:.2e} "
          f"(relative: {relative_difference:.2e})")
    print(f"Same predicted class for all test inputs: {same_predictions}")
    if relative_difference > 1e-3 or not same_predictions:
        print("WARNING: the ONNX model does not match the PyTorch model closely.")
    else:
        print("Numerical check: passed")


def main() -> None:
    set_seed()
    ensure_dirs()
    require_file(config.FP32_PATH, "Run train_baseline.py first.")
    export_model(load_model("fp32"), config.ONNX_FP32_PATH)
    verify_export("fp32", config.ONNX_FP32_PATH)


if __name__ == "__main__":
    main()
