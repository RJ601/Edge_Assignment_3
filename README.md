# Assignment 3: Edge Computing & Intelligent Systems
ResNet-18 on CIFAR-10. Question 1 = quantization (PTQ, QAT, ONNX INT8).
Question 2 = pruning (unstructured, structured, ONNX export).

## Files
| File | Purpose |
|------|---------|
| `config.py` | All paths and settings (one place to change anything) |
| `common.py` | Data loaders, ResNet-18, fusion, model loading (all models) |
| **Question 1** | |
| `train_baseline.py` | Train and save the FP32 baseline (also used by Question 2) |
| `ptq.py` | Post-Training Static Quantization (fuse, calibrate, convert) |
| `qat.py` | Quantization-Aware Training (fake-quant, fine-tune, convert) |
| `export_onnx.py` | Export FP32 model to ONNX and verify it matches PyTorch |
| `quantize_onnx.py` | Create the INT8 ONNX model from the FP32 ONNX model |
| `benchmark.py` | Evaluate the 5 Question 1 models, write Excel + CSV |
| **Question 2** | |
| `prune_utils.py` | Shared helpers: parameter counts, sparsity, fine-tuning |
| `prune_unstructured.py` | Iterative global magnitude pruning, masks, make permanent, fine-tune |
| `prune_structured.py` | Channel pruning, rebuild thinner network, verify, fine-tune |
| `export_onnx_pruned.py` | Export original + both pruned models to ONNX |
| `benchmark_pruning.py` | Evaluate the 6 Question 2 models, write Excel + CSV |
| **Both** | |
| `inference.py` | Classify one image with ANY model (all on CPU) |
| `Colab_Runner.ipynb` | Runs everything on Google Colab |

## Installation
Colab already has PyTorch. Otherwise install PyTorch and torchvision,
then: `pip install -r requirements.txt`

## Run order
```
# ---- Question 1 ----
python train_baseline.py     # needs a GPU for speed; CIFAR-10 downloads automatically
python ptq.py
python qat.py
python export_onnx.py
python quantize_onnx.py
python benchmark.py

# ---- Question 2 (needs the baseline from train_baseline.py) ----
python prune_unstructured.py
python prune_structured.py
python export_onnx_pruned.py
python benchmark_pruning.py

# ---- Inference on one image (any model) ----
python inference.py --model structured --image some_image.png
```
`--model` can be: `fp32`, `ptq`, `qat`, `unstructured`, `structured`,
`onnx_fp32`, `onnx_int8`, `onnx_unstructured`, `onnx_structured`.

## Outputs (all inside `outputs/`)
```
outputs/
  checkpoints/  resnet18_fp32.pth, resnet18_ptq_int8.pth, resnet18_qat_int8.pth,
                resnet18_unstructured_pruned.pth, resnet18_structured_pruned.pth
  onnx/         resnet18_fp32.onnx, resnet18_int8.onnx,
                resnet18_unstructured_pruned.onnx, resnet18_structured_pruned.onnx
  results/      benchmark_results.xlsx / .csv, test_predictions.csv            (Question 1)
                pruning_benchmark_results.xlsx / .csv, pruning_test_predictions.csv (Question 2)
```

## Notes
* Only training and fine-tuning use the GPU. Benchmarking and inference run on the CPU.
* The INT8 ONNX model is created by ONNX Runtime's quantizer from the FP32 ONNX file.
  It is not converted from the PyTorch PTQ/QAT checkpoints.
* Structured pruning removes channels INSIDE each residual block (between its two convs),
  so the block output width, and therefore the residual addition, is unchanged.
* The unstructured-pruned model has the same shape and file size as the original.
  Its benefit is the number of zero weights (see `params_zero_percent` in the results),
  not speed or file size on a normal CPU.
* Latency = average ms per image at batch size 1. Throughput = images/second at batch size 128.
