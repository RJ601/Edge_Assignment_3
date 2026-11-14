"""
benchmark_pruning.py
====================
Question 2, part 6: automatically evaluate the original model and both
pruned models, in PyTorch and in ONNX, on the full CIFAR-10 test set:

    FP32 (PyTorch)              FP32 (ONNX)
    Unstructured-pruned (PyTorch)   Unstructured-pruned (ONNX)
    Structured-pruned (PyTorch)     Structured-pruned (ONNX)

Metrics for every model (same definitions as Question 1's benchmark):
  * Accuracy, macro Precision, macro Recall, macro F1
  * File size (MB), latency (ms per image, batch size 1), throughput (images/s)
  * Parameters: total, non-zero, and the % that are zero

Everything is written into outputs/results/:
  * pruning_benchmark_results.xlsx : workbook with five sheets
        - Summary               one row per model
        - Per_Class_Metrics     precision / recall / F1 for each class
        - Predictions           true vs predicted label for every test image
        - Unstructured_Layers   sparsity of every layer of the sparse model
        - Structured_Channels   inner channel counts before/after pruning
  * pruning_benchmark_results.csv : same as the Summary sheet
  * pruning_test_predictions.csv  : same as the Predictions sheet

Run:  python benchmark_pruning.py
"""

import os

import pandas as pd

import config
from common import (MODEL_REGISTRY, Q2_KINDS, load_model, get_test_loader,
                    set_seed, ensure_dirs)
from benchmark import (measure_throughput, measure_latency,
                       classification_metrics, autofit_columns)
from prune_utils import count_parameters, layer_sparsity_rows, channel_rows

# An ONNX model has the same parameters as the PyTorch model it came from
PYTORCH_TWIN = {
    "onnx_fp32": "fp32",
    "onnx_unstructured": "unstructured",
    "onnx_structured": "structured",
}


def main() -> None:
    set_seed()
    ensure_dirs()

    # Stop early with a clear message if any model file has not been created yet
    missing = [MODEL_REGISTRY[kind][1] for kind in Q2_KINDS
               if not os.path.exists(MODEL_REGISTRY[kind][1])]
    if missing:
        raise FileNotFoundError(
            "These model files are missing. Run the earlier scripts first "
            "(train_baseline, prune_unstructured, prune_structured, "
            "export_onnx_pruned):\n  " + "\n  ".join(missing))

    loader = get_test_loader()
    dataset = loader.dataset  # used for the batch-size-1 latency test

    # Parameter counts for the three PyTorch models (ONNX rows reuse them)
    counts = {kind: count_parameters(load_model(kind))
              for kind in ("fp32", "unstructured", "structured")}

    summary_rows, class_rows, prediction_rows = [], [], []

    for kind in Q2_KINDS:
        name, path = MODEL_REGISTRY[kind]
        print(f"Benchmarking {name}...")
        model = load_model(kind)

        throughput, y_true, y_pred = measure_throughput(model, loader)
        latency_ms = measure_latency(model, dataset)
        metrics, per_class = classification_metrics(y_true, y_pred)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total, nonzero = counts[PYTORCH_TWIN.get(kind, kind)]

        # ---- Summary sheet: one row per model ----
        summary_rows.append({
            "model": name,
            **metrics,
            "size_mb": size_mb,
            "latency_ms_per_image": latency_ms,
            "throughput_images_per_sec": throughput,
            "params_total": total,
            "params_nonzero": nonzero,
            "params_zero_percent": 100.0 * (total - nonzero) / total,
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
    layers_df = pd.DataFrame(layer_sparsity_rows(load_model("unstructured"))).round(2)
    channels_df = pd.DataFrame(
        channel_rows(load_model("fp32"), load_model("structured"))).round(2)

    # Write the Excel workbook (five sheets)
    with pd.ExcelWriter(config.PRUNE_EXCEL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        class_df.to_excel(writer, sheet_name="Per_Class_Metrics", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        layers_df.to_excel(writer, sheet_name="Unstructured_Layers", index=False)
        channels_df.to_excel(writer, sheet_name="Structured_Channels", index=False)
        autofit_columns(writer)

    # Write the CSV files
    summary_df.to_csv(config.PRUNE_SUMMARY_CSV, index=False)
    predictions_df.to_csv(config.PRUNE_PREDICTIONS_CSV, index=False)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved: {config.PRUNE_EXCEL_PATH}")
    print(f"Saved: {config.PRUNE_SUMMARY_CSV}")
    print(f"Saved: {config.PRUNE_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
