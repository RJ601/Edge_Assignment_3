"""
common.py
=========
Shared building blocks used by every script in this assignment:

  1. Reproducibility + folder helpers    (set_seed, ensure_dirs, require_file)
  2. CIFAR-10 data loaders               (get_train_loader, get_test_loader,
                                          get_calibration_loader)
  3. A quantization-friendly ResNet-18   (ResNet18CIFAR)
  4. Layer fusion helper                 (fuse_model)
  5. Evaluation helper                   (evaluate)
  6. ONNX Runtime wrapper                (ONNXModel)
  7. One loader for EVERY model          (MODEL_REGISTRY, Q1_KINDS, Q2_KINDS,
                                          load_model)

Why a custom ResNet-18?
-----------------------
Eager-mode PyTorch quantization needs two things that the standard model
does not give us:
  * QuantStub / DeQuantStub at the model input/output, which mark where
    floating-point tensors become INT8 and back again.
  * FloatFunctional for the residual "+" operation, because a plain "+"
    cannot be quantized in eager mode.
This version also uses a 3x3 stride-1 first convolution and no max-pool,
which suits the small 32x32 CIFAR-10 images.
"""

import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    fuse_modules,
    fuse_modules_qat,
    get_default_qconfig,
    prepare,
    convert,
)
from torch.utils.data import DataLoader, Subset

import config


# ==========================================================================
# 1. Reproducibility and folders
# ==========================================================================
def set_seed(seed: int = config.SEED) -> None:
    """Fix all random number generators so results are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create the output folder and its sub-folders if they do not exist."""
    for folder in (config.CHECKPOINT_DIR, config.ONNX_DIR, config.RESULTS_DIR):
        os.makedirs(folder, exist_ok=True)


def require_file(path: str, hint: str) -> None:
    """Stop with a clear message if an input file from an earlier step is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {hint}")


# ==========================================================================
# 2. Data loading
# ==========================================================================
def _test_transform() -> T.Compose:
    """Transform with NO augmentation: just tensor conversion + normalisation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])


def _train_transform() -> T.Compose:
    """Transform WITH augmentation (random crop + horizontal flip) for training."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])


def get_train_loader() -> DataLoader:
    """Return the augmented CIFAR-10 training loader (50,000 images)."""
    dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True,
        transform=_train_transform(),
    )
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                      num_workers=config.NUM_WORKERS)


def get_test_loader() -> DataLoader:
    """Return the CIFAR-10 test loader (10,000 images, no shuffling)."""
    dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True,
        transform=_test_transform(),
    )
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)


def get_calibration_loader() -> DataLoader:
    """
    Return a loader over a random SUBSET of the training data.

    Calibration (for PTQ and for ONNX INT8) only needs to *observe* typical
    activation ranges, so a small subset is enough. We use the
    no-augmentation transform so the observed ranges match what the model
    sees at inference time.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True,
        transform=_test_transform(),
    )
    # Pick random indices with a fixed seed so the subset is reproducible
    rng = np.random.RandomState(config.SEED)
    indices = rng.choice(len(dataset), config.CALIBRATION_SAMPLES, replace=False)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)


# ==========================================================================
# 3. Quantization-friendly ResNet-18
# ==========================================================================
class BasicBlock(nn.Module):
    """
    One ResNet basic block:  x -> conv-bn-relu -> conv-bn -> (+ shortcut) -> relu

    The "+ shortcut" and the final ReLU are merged into one quantizable
    operation using FloatFunctional.add_relu().

    `mid_planes` is the number of channels BETWEEN the two convolutions.
    Normally it equals `planes`. Structured pruning (Question 2) shrinks it,
    which makes conv1 thinner and conv2 narrower in its input, while the
    block's output width (`planes`) stays the same so the residual "+" still
    has matching shapes.
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 mid_planes: int = None):
        super().__init__()
        if mid_planes is None:
            mid_planes = planes  # un-pruned block
        # Main path
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut path: identity, or a 1x1 conv when the shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        # Quantizable replacement for "out + shortcut" followed by ReLU
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))            # conv -> bn -> relu
        out = self.bn2(self.conv2(out))                      # conv -> bn
        out = self.skip_add.add_relu(out, self.shortcut(x))  # add + relu
        return out


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32x32 CIFAR-10 images.

    QuantStub at the start converts the FP32 input to INT8 (after
    quantization); DeQuantStub at the end converts the INT8 output back to
    FP32. Before quantization, both stubs do nothing (identity).

    `mid_channels` (optional) is a list of 8 numbers: the inner channel count
    of each of the 8 residual blocks, in order. Leave it as None for the
    normal ResNet-18. Structured pruning passes smaller numbers here to
    build the thinner network.
    """

    # Inner channel count of the 8 blocks in the standard (un-pruned) ResNet-18
    DEFAULT_MID_CHANNELS = [64, 64, 128, 128, 256, 256, 512, 512]

    def __init__(self, num_classes: int = 10, mid_channels: list = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = self.DEFAULT_MID_CHANNELS
        if len(mid_channels) != 8:
            raise ValueError("mid_channels must have exactly 8 numbers "
                             "(one per residual block).")
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Stem: a single 3x3 conv (no max-pool, since images are tiny)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Four stages of two BasicBlocks each (64, 128, 256, 512 channels)
        # (each stage takes the 2 inner-channel numbers of its two blocks)
        self.layer1 = self._make_stage(64, 64, 1, mid_channels[0:2])
        self.layer2 = self._make_stage(64, 128, 2, mid_channels[2:4])
        self.layer3 = self._make_stage(128, 256, 2, mid_channels[4:6])
        self.layer4 = self._make_stage(256, 512, 2, mid_channels[6:8])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_planes: int, planes: int, stride: int,
                    mid_channels: list) -> nn.Sequential:
        """Build one stage: first block may downsample, second keeps shape."""
        return nn.Sequential(
            BasicBlock(in_planes, planes, stride, mid_channels[0]),
            BasicBlock(planes, planes, 1, mid_channels[1]),
        )

    def forward(self, x):
        x = self.quant(x)                            # FP32 -> INT8 (after convert)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)                          # INT8 -> FP32 (after convert)
        return x


# ==========================================================================
# 4. Layer fusion
# ==========================================================================
def fuse_model(model: ResNet18CIFAR, qat: bool = False) -> None:
    """
    Fuse Conv + BatchNorm (+ ReLU) into single modules, in place.

    Fusion matters for quantization because the fused module is quantized as
    ONE operation, which is faster and more accurate than quantizing the
    conv, bn and relu separately.

    Args:
        model: the ResNet18CIFAR to fuse.
        qat:   False for PTQ (model must be in eval mode, uses fuse_modules).
               True  for QAT (model must be in train mode, uses fuse_modules_qat).
    """
    fuse_fn = fuse_modules_qat if qat else fuse_modules

    # Stem: conv1 + bn1 + relu
    fuse_fn(model, [["conv1", "bn1", "relu"]], inplace=True)

    # Every BasicBlock: (conv1, bn1, relu1) and (conv2, bn2)
    for module in model.modules():
        if isinstance(module, BasicBlock):
            fuse_fn(module, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                    inplace=True)
            # If the block has a 1x1 conv shortcut, fuse its conv + bn too
            if len(module.shortcut) > 0:
                fuse_fn(module.shortcut, [["0", "1"]], inplace=True)


# ==========================================================================
# 5. Evaluation
# ==========================================================================
def evaluate(model, loader: DataLoader, device: str = "cpu") -> float:
    """
    Compute top-1 accuracy (%) of `model` over every batch in `loader`.

    Quantized INT8 models only run on CPU, so use device="cpu" for them.
    Works for PyTorch models and for the ONNXModel wrapper below.
    """
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ==========================================================================
# 6. ONNX Runtime wrapper
# ==========================================================================
class ONNXModel:
    """
    Makes an ONNX Runtime session behave like a PyTorch model.

    After wrapping, you can call  logits = model(images)  with a torch tensor
    and get a torch tensor back. This lets the inference and benchmark code
    treat PyTorch and ONNX models in exactly the same way.

    The session is forced onto the CPU (CPUExecutionProvider), because the
    assignment asks for CPU inference and because it keeps the speed
    comparison with the PyTorch INT8 models fair.
    """

    def __init__(self, path: str):
        # Imported here so PyTorch-only scripts do not need onnxruntime
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def eval(self):
        """Present for API compatibility with torch models; does nothing."""
        return self

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Run a (N, 3, 32, 32) float tensor and return (N, 10) logits."""
        # ONNX Runtime works on NumPy arrays, so convert in and out
        outputs = self.session.run(None, {self.input_name: images.numpy()})[0]
        return torch.from_numpy(outputs)


# ==========================================================================
# 7. Loading any model
# ==========================================================================
# kind -> (name shown in tables, file path)
MODEL_REGISTRY = {
    "fp32":             ("FP32 (PyTorch)",               config.FP32_PATH),
    "ptq":              ("PTQ INT8 (PyTorch)",           config.PTQ_PATH),
    "qat":              ("QAT INT8 (PyTorch)",           config.QAT_PATH),
    "onnx_fp32":        ("FP32 (ONNX)",                  config.ONNX_FP32_PATH),
    "onnx_int8":        ("INT8 (ONNX)",                  config.ONNX_INT8_PATH),
    "unstructured":     ("Unstructured-pruned (PyTorch)", config.UNSTRUCTURED_PATH),
    "structured":       ("Structured-pruned (PyTorch)",   config.STRUCTURED_PATH),
    "onnx_unstructured": ("Unstructured-pruned (ONNX)",   config.ONNX_UNSTRUCTURED_PATH),
    "onnx_structured":  ("Structured-pruned (ONNX)",      config.ONNX_STRUCTURED_PATH),
}

# Which models each question's benchmark evaluates
Q1_KINDS = ["fp32", "ptq", "qat", "onnx_fp32", "onnx_int8"]
Q2_KINDS = ["fp32", "unstructured", "structured",
            "onnx_fp32", "onnx_unstructured", "onnx_structured"]


def load_model(kind: str):
    """
    Load a trained model by name. Always returns an eval-mode CPU model.

    Args:
        kind: one of the keys in MODEL_REGISTRY, for example
              "fp32", "ptq", "qat", "unstructured", "structured",
              "onnx_fp32", "onnx_int8", "onnx_unstructured", "onnx_structured".

    How each kind is loaded:
        * ONNX kinds          : opened with ONNX Runtime (CPU).
        * fp32, unstructured  : a normal ResNet-18; the unstructured-pruned
                                model has the same shape as the original, only
                                many of its weights are zero.
        * structured          : the checkpoint also stores the reduced channel
                                counts, so the thinner network is rebuilt first.
        * ptq, qat            : A quantized model's weights cannot be loaded
                                into a plain ResNet, because its layers are
                                different. So we first rebuild the same
                                *structure* (fuse -> prepare -> convert) and
                                then copy the saved INT8 weights and scales
                                into it. PTQ and QAT end up with the same
                                structure, so both use the same rebuild code.
    """
    if kind not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model kind '{kind}'. Use one of: {list(MODEL_REGISTRY)}")
    path = MODEL_REGISTRY[kind][1]

    # ---- ONNX models: just open them with ONNX Runtime -------------------
    if kind.startswith("onnx"):
        return ONNXModel(path)

    # ---- PyTorch FP32 model and the unstructured-pruned model ------------
    if kind in ("fp32", "unstructured"):
        model = ResNet18CIFAR()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()

    # ---- Structured-pruned model: rebuild the thinner network first ------
    if kind == "structured":
        checkpoint = torch.load(path, map_location="cpu")
        model = ResNet18CIFAR(mid_channels=checkpoint["mid_channels"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    # ---- PyTorch INT8 models (PTQ or QAT) --------------------------------
    torch.backends.quantized.engine = config.QUANT_BACKEND

    # Rebuild the quantized structure
    model = ResNet18CIFAR().eval()
    fuse_model(model, qat=False)
    model.qconfig = get_default_qconfig(config.QUANT_BACKEND)
    prepare(model, inplace=True)

    # convert() may warn that the observers saw no data; that is harmless
    # here because the real scales are loaded from the file just below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        convert(model, inplace=True)

    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()
