"""
inference.py
============
Step 6 of Question 1 (and the inference pipeline of Question 2): one script
that works with EVERY model in this assignment, all on the CPU:

    fp32               FP32 PyTorch (the original model)
    ptq                PTQ INT8 PyTorch
    qat                QAT INT8 PyTorch
    unstructured       Unstructured-pruned PyTorch
    structured         Structured-pruned PyTorch
    onnx_fp32          FP32 ONNX         (through ONNX Runtime)
    onnx_int8          INT8 ONNX         (through ONNX Runtime)
    onnx_unstructured  Unstructured-pruned ONNX
    onnx_structured    Structured-pruned ONNX

It classifies a single image file and prints the predicted class with its
confidence. Other scripts can also import `predict_image` directly.

Run:
    python inference.py --model fp32      --image cat.png
    python inference.py --model qat       --image cat.png
    python inference.py --model onnx_int8 --image cat.png
    python inference.py --model structured --image cat.png
"""

import argparse

import torch
import torchvision.transforms as T
from PIL import Image

import config
from common import load_model, MODEL_REGISTRY


def preprocess(image_path: str) -> torch.Tensor:
    """
    Load an image file and turn it into a model-ready tensor.

    Steps: force RGB -> resize to 32x32 (CIFAR-10 size) -> to tensor ->
    normalise with the CIFAR-10 mean/std -> add a batch dimension.
    """
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1, 3, 32, 32)


def predict_image(model, image_path: str):
    """
    Run one image through `model` (a PyTorch model or an ONNXModel).

    Returns:
        (class_name, confidence) where confidence is a probability in [0, 1].
    """
    batch = preprocess(image_path)
    with torch.inference_mode():
        logits = model(batch)                      # raw scores, shape (1, 10)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)
    return config.CIFAR10_CLASSES[index.item()], confidence.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 inference")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY),
                        help="Which model to use")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    model = load_model(args.model)
    label, confidence = predict_image(model, args.image)
    name = MODEL_REGISTRY[args.model][0]
    print(f"Model: {name} | Prediction: {label} "
          f"({confidence * 100:.1f}% confidence)")


if __name__ == "__main__":
    main()
