"""
benchmark.py
============
Step 7 of Question 1: automatically evaluate ALL five models on the full
CIFAR-10 test set (10,000 images):

    FP32 (PyTorch) | PTQ INT8 (PyTorch) | QAT INT8 (PyTorch)
    FP32 (ONNX)    | INT8 (ONNX)

Metrics logged for every model:
  * Accuracy, macro Precision, macro Recall, macro F1   (quality)
  * Model file size in MB                               (compression)
  * Latency  = average ms to classify ONE image (batch size 1)
  * Throughput = images per second when processing batches

Everything is written into the single output folder (outputs/results/):
  * benchmark_results.xlsx : one workbook with three sheets
        - Summary            one row per model
        - Per_Class_Metrics  precision / recall / F1 for each class
        - Predictions        every test image's true vs predicted label
  * benchmark_results.csv  : same as the Summary sheet
  * test_predictions.csv   : same as the Predictions sheet

All models run on the CPU so the speed comparison is fair (INT8 models can
only run on the CPU).

Run:  python benchmark.py
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

import config
from common import (MODEL_REGISTRY, Q1_KINDS, load_model, get_test_loader,
                    set_seed, ensure_dirs)


# ==========================================================================
# Measurement helpers
# ==========================================================================
def measure_throughput(model, loader):
    """
    Run `model` over the whole test loader and time only the model calls.

    Returns:
        throughput (images/second), y_true (list), y_pred (list)
    """
    # Warm-up: a few untimed passes so one-time start-up costs are excluded
    with torch.inference_mode():
        for i, (images, _) in enumerate(loader):
            if i >= config.WARMUP_BATCHES:
                break
            model(images)

    y_true, y_pred = [], []
    total_time = 0.0  # seconds spent inside model(...) only

    with torch.inference_mode():
        for images, labels in loader:
            start = time.perf_counter()
            outputs = model(images)                      # <- the timed part
            total_time += time.perf_counter() - start

            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).tolist())

    throughput = len(y_true) / total_time
    return throughput, y_true, y_pred


def measure_latency(model, dataset) -> float:
    """
    Average time (ms) to classify ONE image, with batch size 1.

    This mimics real use, where images arrive one at a time. The first
    LATENCY_WARMUP runs are not timed; the next LATENCY_SAMPLES are.
    """
    total_runs = config.LATENCY_WARMUP + config.LATENCY_SAMPLES
    times = []
    with torch.inference_mode():
        for i in range(total_runs):
            image = dataset[i][0].unsqueeze(0)           # shape (1, 3, 32, 32)
            start = time.perf_counter()
            model(image)                                 # <- the timed part
            elapsed = time.perf_counter() - start
            if i >= config.LATENCY_WARMUP:
                times.append(elapsed)
    return 1000.0 * float(np.mean(times))


def classification_metrics(y_true, y_pred):
    """
    Compute quality metrics from the true and predicted labels.

    "Macro" averaging computes the metric for each of the 10 classes and
    then takes the plain average, so every class counts equally.

    Returns:
        summary: dict with accuracy, macro precision, macro recall, macro F1 (in %)
        per_class: tuple (precision, recall, f1, support), each an array of 10 values
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = list(range(len(config.CIFAR10_CLASSES)))

    accuracy = 100.0 * float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)

    summary = {
        "accuracy_percent": accuracy,
        "precision_macro_percent": 100.0 * precision,
        "recall_macro_percent": 100.0 * recall,
        "f1_macro_percent": 100.0 * f1,
    }
    return summary, per_class


# ==========================================================================
# Excel formatting
# ==========================================================================
def autofit_columns(writer) -> None:
    """Widen every column to fit its longest value and freeze the header row."""
    for sheet in writer.sheets.values():
        for column in sheet.columns:
            longest = max(len(str(cell.value)) if cell.value is not None else 0
                          for cell in column)
            sheet.column_dimensions[column[0].column_letter].width = longest + 3
        sheet.freeze_panes = "A2"  # keep the header visible while scrolling


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    set_seed()
    ensure_dirs()

    # Stop early with a clear message if any model file has not been created yet
    missing = [MODEL_REGISTRY[kind][1] for kind in Q1_KINDS
               if not os.path.exists(MODEL_REGISTRY[kind][1])]
    if missing:
        raise FileNotFoundError(
            "These model files are missing. Run the earlier scripts first:\n  "
            + "\n  ".join(missing))

    loader = get_test_loader()
    dataset = loader.dataset  # used for the batch-size-1 latency test

    summary_rows, class_rows, prediction_rows = [], [], []

    for kind in Q1_KINDS:
        name, path = MODEL_REGISTRY[kind]
        print(f"Benchmarking {name}...")
        model = load_model(kind)

        throughput, y_true, y_pred = measure_throughput(model, loader)
        latency_ms = measure_latency(model, dataset)
        metrics, per_class = classification_metrics(y_true, y_pred)
        size_mb = os.path.getsize(path) / (1024 * 1024)

        # ---- Summary sheet: one row per model ----
        summary_rows.append({
            "model": name,
            **metrics,
            "size_mb": size_mb,
            "latency_ms_per_image": latency_ms,
            "throughput_images_per_sec": throughput,
        })

        # ---- Per-class sheet: one row per (model, class) ----
        precision, recall, f1, support = per_class
        for i, class_name in enumerate(config.CIFAR10_CLASSES):
            class_rows.append({
                "model": name,
                "class": class_name,
                "precision_percent": 100.0 * precision[i],
                "recall_percent": 100.0 * recall[i],
                "f1_percent": 100.0 * f1[i],
                "support": int(support[i]),
            })

        # ---- Predictions sheet: one row per (model, test image) ----
        for index, (true, pred) in enumerate(zip(y_true, y_pred)):
            prediction_rows.append({
                "model": name,
                "image_index": index,
                "true_label": config.CIFAR10_CLASSES[true],
                "predicted_label": config.CIFAR10_CLASSES[pred],
                "correct": int(true == pred),
            })

        print(f"  accuracy {metrics['accuracy_percent']:.2f}% | "
              f"F1 {metrics['f1_macro_percent']:.2f}% | size {size_mb:.2f} MB | "
              f"latency {latency_ms:.3f} ms | {throughput:.1f} images/s")

    # Round the numbers so the tables are easy to read
    summary_df = pd.DataFrame(summary_rows).round(3)
    class_df = pd.DataFrame(class_rows).round(2)
    predictions_df = pd.DataFrame(prediction_rows)

    # Write the Excel workbook (three sheets)
    with pd.ExcelWriter(config.EXCEL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        class_df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        autofit_columns(writer)

    # Write the CSV files required by the assignment
    summary_df.to_csv(config.SUMMARY_CSV, index=False)
    predictions_df.to_csv(config.PREDICTIONS_CSV, index=False)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved: {config.EXCEL_PATH}")
    print(f"Saved: {config.SUMMARY_CSV}")
    print(f"Saved: {config.PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
