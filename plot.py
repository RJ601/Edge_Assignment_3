# =============================================================================
# Compare ALL model variants (baseline, quantization, pruning; PyTorch + ONNX)
# on every logged metric and save the final plots to outputs/results
# =============================================================================
import os                                   # folder creation / path joining
import re                                   # used to make safe file names
import numpy as np                          # numeric arrays / NaN handling
import pandas as pd                         # reading CSVs, merging tables
import matplotlib.pyplot as plt             # all plotting

# ----------------------------- 1. CONFIGURATION ------------------------------
OUT_DIR = os.path.join("outputs", "results")  # plots are saved here
os.makedirs(OUT_DIR, exist_ok=True)           # create the folder if it doesn't exist
DATA_DIR = OUT_DIR                            # the CSVs are in the same folder

QUANT_CSV = os.path.join(DATA_DIR, "benchmark_results.csv")
PRUNE_CSV = os.path.join(DATA_DIR, "pruning_benchmark_results.csv")
QUANT_PRED = os.path.join(DATA_DIR, "test_predictions.csv")
PRUNE_PRED = os.path.join(DATA_DIR, "pruning_test_predictions.csv")

# Every metric that is logged: (column name, axis label, is higher better?)
# "higher better" decides which colour means "good" in the comparison heatmaps.
METRICS = [
    ("accuracy_percent",          "Accuracy (%)",              True),
    ("precision_macro_percent",   "Precision, macro (%)",      True),
    ("recall_macro_percent",      "Recall, macro (%)",         True),
    ("f1_macro_percent",          "F1, macro (%)",             True),
    ("size_mb",                   "Model size (MB)",           False),
    ("latency_ms_per_image",      "Latency (ms / image)",      False),
    ("throughput_images_per_sec", "Throughput (images / s)",   True),
    ("params_total",              "Total parameters",          False),
    ("params_nonzero",            "Non-zero parameters",       False),
    ("params_zero_percent",       "Zero parameters (%)",       True),
]
# Columns that are classification-quality scores (their axis gets zoomed in,
# otherwise 92.8 vs 93.2 would look identical on a 0-100 axis).
SCORE_COLS = {"accuracy_percent", "precision_macro_percent",
              "recall_macro_percent", "f1_macro_percent"}

# ----------------------------- 2. LOAD + MERGE -------------------------------
quant = pd.read_csv(QUANT_CSV)              # quantization benchmark table
prune = pd.read_csv(PRUNE_CSV)              # pruning benchmark table

# Both tables contain "FP32 (PyTorch)" and "FP32 (ONNX)". Their accuracy/F1 are
# identical, but latency/throughput differ slightly because they come from two
# separate benchmark runs. Stacking the tables and averaging rows with the same
# model name gives ONE row per variant (sort=False keeps the original order).
df = (pd.concat([quant, prune], ignore_index=True)
        .groupby("model", sort=False).mean(numeric_only=True))
MODELS = list(df.index)                     # ordered list of all unique variants

# Only plot metrics that actually exist in the data.
METRICS = [m for m in METRICS if m[0] in df.columns]

# One colour per variant; ONNX variants additionally get a hatch pattern so
# PyTorch-vs-ONNX can be told apart without reading the labels.
cmap = plt.get_cmap("tab10")
COLORS = {m: cmap(i % 10) for i, m in enumerate(MODELS)}
HATCH = {m: ("//" if "ONNX" in m else "") for m in MODELS}

def short(name):
    """Wrap a model name onto two lines so x-axis labels don't overlap."""
    return name.replace(" (", "\n(")

def save(fig, filename):
    """Save a figure into OUT_DIR at high resolution and free its memory."""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", path)

# ------------------- 3. FIGURE 1: every metric, every model ------------------
# One bar chart per metric, all variants side by side.
ncols = 3
nrows = int(np.ceil(len(METRICS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.2 * nrows))
axes = np.array(axes).ravel()
for ax, (col, label, _) in zip(axes, METRICS):
    vals = df[col].values.astype(float)     # NaN where a model has no value
    bars = ax.bar(range(len(MODELS)), vals,
                  color=[COLORS[m] for m in MODELS],
                  hatch=None, edgecolor="black", linewidth=0.6)
    for b, m in zip(bars, MODELS):          # apply ONNX hatch bar by bar
        b.set_hatch(HATCH[m])
    # Write the value above each bar (skip models without that metric).
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:,.3f}" if v < 1000 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=7)
    # Zoom the y-axis for score metrics so small differences become visible.
    if col in SCORE_COLS:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = max((hi - lo) * 0.5, 0.2)
        ax.set_ylim(lo - pad, hi + pad)
        label += "  [zoomed axis]"
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([short(m) for m in MODELS], rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if df[col].isna().any():                # e.g. parameter counts are logged only for the pruning runs
        ax.set_xlabel("(missing bars = not logged for that model)", fontsize=7, style="italic")
for ax in axes[len(METRICS):]:              # hide unused panels
    ax.axis("off")
fig.suptitle("All model variants - every logged metric", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])      # leave room for the main title
save(fig, "01_all_metrics_bars.png")

# ---------- 4. FIGURE 2: one-glance scorecard (best = green, worst = red) -----
# Rows = models, columns = metrics. Colour = min-max normalised score where
# 1 (green) is the best model for that metric. Text = the raw value.
core = [m for m in METRICS if not df[m[0]].isna().any()]   # metrics logged for every model
norm = pd.DataFrame(index=MODELS)
for col, label, higher in core:
    v = df[col]
    rng = v.max() - v.min()
    n = (v - v.min()) / rng if rng > 0 else v * 0 + 1.0    # all-equal column -> all 1
    norm[col] = n if higher else 1 - n                      # flip so 1 is always "best"
fig, ax = plt.subplots(figsize=(1.6 * len(core) + 3, 0.6 * len(MODELS) + 2.5))
im = ax.imshow(norm.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i, m in enumerate(MODELS):
    for j, (col, _, _) in enumerate(core):
        ax.text(j, i, f"{df.loc[m, col]:,.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(core)))
ax.set_xticklabels([c[1] for c in core], rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=9)
ax.set_title("Scorecard: green = best model for that metric, red = worst",
             fontweight="bold")
fig.colorbar(im, ax=ax, label="normalised score (1 = best)")
fig.tight_layout()
save(fig, "02_scorecard.png")

# -------- 5. FIGURE 3: trade-off scatter plots (quality vs. efficiency) ------
# Bubble area is proportional to model size, so three dimensions fit in 2-D.
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
pairs = [("latency_ms_per_image", "Latency (ms / image)  - lower is better"),
         ("throughput_images_per_sec", "Throughput (images / s)  - higher is better"),
         ("size_mb", "Model size (MB)  - lower is better")]
for ax, (xcol, xlabel) in zip(axes, pairs):
    for m in MODELS:
        ax.scatter(df.loc[m, xcol], df.loc[m, "accuracy_percent"],
                   s=df.loc[m, "size_mb"] * 12 + 40,    # bubble size = model size
                   color=COLORS[m], edgecolor="black", alpha=0.85,
                   hatch=HATCH[m], label=m)
        ax.annotate(short(m).replace("\n", " "),
                    (df.loc[m, xcol], df.loc[m, "accuracy_percent"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="best", title="bubble size = model size", markerscale=0.4)
fig.suptitle("Accuracy vs. efficiency trade-offs", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "03_tradeoff_scatter.png")

# ------ 6. FIGURE 4+: every model vs. every other model, per metric ----------
# For each metric an N x N heatmap. Cell (row i, column j) answers:
#   "How much better/worse is row-model i than column-model j?"
# shown as relative change in %  =  (value_i - value_j) / value_j * 100.
# Colour: green = row model is BETTER (taking "higher/lower is better" into
# account), red = row model is WORSE, white = equal.
for col, label, higher in METRICS:
    v = df[col].dropna()                    # only models that have this metric
    if len(v) < 2:
        continue                            # nothing to compare
    names = list(v.index)
    n = len(names)
    rel = np.zeros((n, n))                  # relative change in %
    for i in range(n):
        for j in range(n):
            base = v.iloc[j]
            rel[i, j] = (v.iloc[i] - base) / base * 100 if base != 0 else np.nan
    score = rel if higher else -rel         # positive = row model better
    lim = np.nanmax(np.abs(score))
    lim = lim if lim > 0 else 1.0           # avoid a zero colour range
    fig, ax = plt.subplots(figsize=(0.95 * n + 4, 0.8 * n + 3))
    im = ax.imshow(score, cmap="RdYlGn", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(rel[i, j]):
                ax.text(j, i, f"{rel[i, j]:+.2f}%", ha="center", va="center", fontsize=7)
            elif i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(x) for x in names], rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("compared against (column)")
    ax.set_ylabel("model (row)")
    ax.set_title(f"{label}\nrelative change of row model vs. column model "
                 f"({'higher' if higher else 'lower'} is better)", fontsize=10,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="green = row better, red = row worse")
    fig.tight_layout()
    save(fig, f"04_pairwise_{re.sub(r'[^a-z0-9]+', '_', col.lower())}.png")

# ------- 7. FIGURES 5-7: per-image predictions (class-level + agreement) -----
# These come from the two *_test_predictions.csv files.
pred = pd.concat([pd.read_csv(QUANT_PRED), pd.read_csv(PRUNE_PRED)], ignore_index=True)
# FP32 rows exist in both files; keep one copy of each (model, image) pair.
pred = pred.drop_duplicates(subset=["model", "image_index"], keep="first")
pmodels = [m for m in MODELS if m in set(pred["model"])]   # keep the same model order

# 7a. Per-class accuracy: rows = models, columns = the 10 classes.
cls_acc = (pred.groupby(["model", "true_label"])["correct"].mean() * 100).unstack()
cls_acc = cls_acc.loc[pmodels]
fig, ax = plt.subplots(figsize=(1.1 * cls_acc.shape[1] + 3, 0.6 * len(pmodels) + 2.5))
im = ax.imshow(cls_acc.values, cmap="viridis", aspect="auto")
for i in range(cls_acc.shape[0]):
    for j in range(cls_acc.shape[1]):
        ax.text(j, i, f"{cls_acc.values[i, j]:.1f}", ha="center", va="center",
                fontsize=8, color="white")
ax.set_xticks(range(cls_acc.shape[1]))
ax.set_xticklabels(cls_acc.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pmodels)))
ax.set_yticklabels(pmodels, fontsize=9)
ax.set_title("Per-class accuracy (%) for every model", fontweight="bold")
fig.colorbar(im, ax=ax, label="accuracy (%)")
fig.tight_layout()
save(fig, "05_per_class_accuracy.png")

# 7b. Per-class accuracy difference vs. the FP32 (PyTorch) baseline.
base = "FP32 (PyTorch)"
if base in cls_acc.index:
    diff = cls_acc.sub(cls_acc.loc[base], axis=1).drop(index=base)
    lim = max(np.abs(diff.values).max(), 0.1)
    fig, ax = plt.subplots(figsize=(1.1 * diff.shape[1] + 3, 0.6 * len(diff) + 2.5))
    im = ax.imshow(diff.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            ax.text(j, i, f"{diff.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(diff.shape[1]))
    ax.set_xticklabels(diff.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(diff)))
    ax.set_yticklabels(diff.index, fontsize=9)
    ax.set_title(f"Per-class accuracy change vs. {base} (percentage points)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="green = better than baseline, red = worse")
    fig.tight_layout()
    save(fig, "06_per_class_accuracy_vs_baseline.png")

# 7c. Prediction agreement: for each pair of models, the % of test images on
# which both predict the SAME class (right or wrong).
wide = pred.pivot(index="image_index", columns="model", values="predicted_label")[pmodels]
agree = np.array([[(wide[a] == wide[b]).mean() * 100 for b in pmodels] for a in pmodels])
fig, ax = plt.subplots(figsize=(0.95 * len(pmodels) + 4, 0.85 * len(pmodels) + 3))
im = ax.imshow(agree, cmap="Blues", vmin=agree.min() - 1, vmax=100)
for i in range(len(pmodels)):
    for j in range(len(pmodels)):
        ax.text(j, i, f"{agree[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if agree[i, j] > (agree.min() + 100) / 2 else "black")
ax.set_xticks(range(len(pmodels)))
ax.set_xticklabels([short(m) for m in pmodels], rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(pmodels)))
ax.set_yticklabels(pmodels, fontsize=8)
ax.set_title("Prediction agreement between every pair of models (% of test images)",
             fontweight="bold")
fig.colorbar(im, ax=ax, label="agreement (%)")
fig.tight_layout()
save(fig, "07_prediction_agreement.png")

print("\nDone. All plots are in:", os.path.abspath(OUT_DIR))