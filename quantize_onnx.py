"""
quantize_onnx.py
================
Step 5 of Question 1: create the INT8 ONNX model from the FP32 ONNX model,
using ONNX Runtime's own static quantization.

NOTE FOR YOUR REPORT: this INT8 model is produced by ONNX Runtime's
quantizer, NOT converted from the PyTorch PTQ/QAT checkpoints. (PyTorch
eager-mode INT8 models cannot be exported to ONNX reliably.)

Pipeline (same idea as PyTorch PTQ):
  1. Take the FP32 ONNX model.
  2. Calibrate: feed the same subset of training images used for PyTorch
     PTQ, so ONNX Runtime can record the activation range at each layer.
  3. Write a new ONNX file whose weights and activations are INT8.

Run:  python quantize_onnx.py
"""

import os

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

import config
from common import get_calibration_loader, set_seed, ensure_dirs


class CifarCalibrationReader(CalibrationDataReader):
    """
    Hands calibration batches to ONNX Runtime, one batch per call.

    ONNX Runtime repeatedly calls get_next(). Each call must return a dict
    {input_name: numpy_array}, and None once the data is finished.
    """

    def __init__(self, loader, input_name: str):
        self.input_name = input_name
        # Convert every batch to a NumPy array up front (1000 images is tiny)
        batches = [images.numpy() for images, _ in loader]
        self.iterator = iter(batches)

    def get_next(self):
        """Return the next calibration batch, or None when there are no more."""
        batch = next(self.iterator, None)
        if batch is None:
            return None
        return {self.input_name: batch}


def main() -> None:
    set_seed()
    ensure_dirs()

    if not os.path.exists(config.ONNX_FP32_PATH):
        raise FileNotFoundError(
            f"{config.ONNX_FP32_PATH} not found. Run export_onnx.py first.")

    print(f"Calibrating with {config.CALIBRATION_SAMPLES} training images...")
    reader = CifarCalibrationReader(get_calibration_loader(),
                                    config.ONNX_INPUT_NAME)

    quantize_static(
        model_input=config.ONNX_FP32_PATH,
        model_output=config.ONNX_INT8_PATH,
        calibration_data_reader=reader,
        # QDQ = the model keeps explicit Quantize/Dequantize nodes. This is
        # the format ONNX Runtime recommends for CPU.
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,   # activations stored as unsigned 8-bit
        weight_type=QuantType.QInt8,        # weights stored as signed 8-bit
        per_channel=True,                   # one scale per output channel (more accurate)
        calibrate_method=CalibrationMethod.MinMax,  # same min/max idea as PyTorch PTQ
    )

    fp32_mb = os.path.getsize(config.ONNX_FP32_PATH) / (1024 * 1024)
    int8_mb = os.path.getsize(config.ONNX_INT8_PATH) / (1024 * 1024)
    print(f"Saved INT8 ONNX model to {config.ONNX_INT8_PATH}")
    print(f"Size: {fp32_mb:.2f} MB (FP32) -> {int8_mb:.2f} MB (INT8)")


if __name__ == "__main__":
    main()
