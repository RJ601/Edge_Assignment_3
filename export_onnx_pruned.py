"""
export_onnx_pruned.py
=====================
Question 2, part 5 (deployment): export the three Question 2 PyTorch models
to ONNX so they can be run on any CPU with ONNX Runtime:

    original FP32          -> outputs/onnx/resnet18_fp32.onnx
    unstructured-pruned    -> outputs/onnx/resnet18_unstructured_pruned.onnx
    structured-pruned      -> outputs/onnx/resnet18_structured_pruned.onnx

It reuses export_model() and verify_export() from export_onnx.py, so every
file is also checked against its PyTorch model.

Run:  python export_onnx_pruned.py
"""

import config
from common import load_model, set_seed, ensure_dirs, require_file
from export_onnx import export_model, verify_export

# (kind, ONNX output path)
EXPORTS = [
    ("fp32", config.ONNX_FP32_PATH),
    ("unstructured", config.ONNX_UNSTRUCTURED_PATH),
    ("structured", config.ONNX_STRUCTURED_PATH),
]


def main() -> None:
    set_seed()
    ensure_dirs()
    require_file(config.FP32_PATH, "Run train_baseline.py first.")
    require_file(config.UNSTRUCTURED_PATH, "Run prune_unstructured.py first.")
    require_file(config.STRUCTURED_PATH, "Run prune_structured.py first.")

    for kind, onnx_path in EXPORTS:
        print(f"\n=== {kind} ===")
        export_model(load_model(kind), onnx_path)
        verify_export(kind, onnx_path)


if __name__ == "__main__":
    main()
