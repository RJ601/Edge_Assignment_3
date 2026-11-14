"""
config.py
=========
Central configuration for the whole assignment.

  Question 1: FP32 baseline -> PTQ INT8 -> QAT INT8 -> ONNX FP32 -> ONNX INT8
  Question 2: FP32 baseline -> unstructured pruning / structured pruning
              -> ONNX export -> benchmark

Every script imports its settings from this file, so changing a value here
changes it everywhere. This is what makes the pipeline reproducible.

ALL generated files (checkpoints, ONNX models, CSV and Excel results) go
inside ONE folder, OUTPUT_DIR. Only the downloaded CIFAR-10 dataset lives
outside it (in DATA_DIR), because it is an input, not an output.
"""

import os

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42  # Fixed random seed so every run gives the same results

# --------------------------------------------------------------------------
# Folder and file paths
# --------------------------------------------------------------------------
DATA_DIR = "./data"        # CIFAR-10 is downloaded here automatically (input)

OUTPUT_DIR = "./outputs"   # THE single output folder for everything generated
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # PyTorch checkpoints
ONNX_DIR = os.path.join(OUTPUT_DIR, "onnx")               # ONNX models
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")         # CSV + Excel results

# PyTorch checkpoints (state_dict files)
FP32_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_fp32.pth")
PTQ_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_ptq_int8.pth")
QAT_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_qat_int8.pth")

# ONNX models
ONNX_FP32_PATH = os.path.join(ONNX_DIR, "resnet18_fp32.onnx")
ONNX_INT8_PATH = os.path.join(ONNX_DIR, "resnet18_int8.onnx")

# Question 2: pruned models (PyTorch checkpoints and ONNX files)
UNSTRUCTURED_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_unstructured_pruned.pth")
STRUCTURED_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_structured_pruned.pth")
ONNX_UNSTRUCTURED_PATH = os.path.join(ONNX_DIR, "resnet18_unstructured_pruned.onnx")
ONNX_STRUCTURED_PATH = os.path.join(ONNX_DIR, "resnet18_structured_pruned.onnx")

# Question 1 result files
SUMMARY_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")     # one row per model
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "test_predictions.csv")  # one row per model+image
EXCEL_PATH = os.path.join(RESULTS_DIR, "benchmark_results.xlsx")     # everything in one workbook

# Question 2 result files
PRUNE_SUMMARY_CSV = os.path.join(RESULTS_DIR, "pruning_benchmark_results.csv")
PRUNE_PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "pruning_test_predictions.csv")
PRUNE_EXCEL_PATH = os.path.join(RESULTS_DIR, "pruning_benchmark_results.xlsx")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # Standard CIFAR-10 channel means
CIFAR10_STD = (0.2470, 0.2435, 0.2616)   # Standard CIFAR-10 channel std devs

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------
BATCH_SIZE = 128            # Batch size for training and evaluation
NUM_WORKERS = 2             # Data loading worker processes

BASELINE_EPOCHS = 30        # Epochs for FP32 baseline training
BASELINE_LR = 0.1           # Starting learning rate for baseline (SGD)
MOMENTUM = 0.9              # SGD momentum
WEIGHT_DECAY = 5e-4         # L2 regularisation strength

CALIBRATION_SAMPLES = 1000  # Training images used to calibrate PTQ and ONNX INT8

QAT_EPOCHS = 5              # Fine-tuning epochs for QAT
QAT_LR = 0.001              # Learning rate for QAT (small, since we fine-tune)

# --------------------------------------------------------------------------
# PyTorch quantization backend
# --------------------------------------------------------------------------
# "fbgemm" is for x86 CPUs (Intel/AMD laptops, servers and Google Colab).
QUANT_BACKEND = "fbgemm"

# --------------------------------------------------------------------------
# ONNX settings
# --------------------------------------------------------------------------
ONNX_OPSET = 17             # ONNX operator-set version used for the export
ONNX_INPUT_NAME = "input"   # Name of the model's input tensor inside the ONNX file
ONNX_OUTPUT_NAME = "logits" # Name of the model's output tensor inside the ONNX file

# --------------------------------------------------------------------------
# Benchmark settings
# --------------------------------------------------------------------------
WARMUP_BATCHES = 3          # Untimed batches before measuring throughput
LATENCY_SAMPLES = 200       # Single images timed one-by-one for latency
LATENCY_WARMUP = 20         # Untimed single-image runs before measuring latency

# --------------------------------------------------------------------------
# Question 2: pruning settings
# --------------------------------------------------------------------------
# Fine-tuning optimiser settings (SGD, uses MOMENTUM and WEIGHT_DECAY above)
FINE_TUNE_LR = 0.01          # Starting learning rate for every pruning fine-tune stage

# Unstructured (magnitude / weight-level) pruning
UNSTRUCT_TARGET_SPARSITY = 0.80   # Final fraction of conv + fc weights that are zero
UNSTRUCT_ITERATIONS = 5           # Prune in this many small steps (iterative pruning)
UNSTRUCT_FT_EPOCHS_PER_ITER = 2   # Fine-tune epochs after EACH pruning step
UNSTRUCT_FINAL_FT_EPOCHS = 5      # Extra fine-tune epochs after the last pruning step

# Structured (channel) pruning
STRUCT_PRUNE_RATIO = 0.50         # Fraction of channels removed inside every residual block
STRUCT_FT_EPOCHS = 10             # Fine-tune epochs for the rebuilt, smaller network
