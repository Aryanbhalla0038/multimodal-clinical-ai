"""Generate publication-ready plots from the 3-config ablation results.

Run:
    python results/ablation/make_plots.py
Outputs:
    results/ablation/plots/macro_auc_bar.png
    results/ablation/plots/per_label_heatmap.png
    results/ablation/plots/training_curves.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ---- Load -------------------------------------------------------------------
with open(HERE / "ablation_full.json") as f:
    full = json.load(f)
per_label = pd.read_csv(HERE / "ablation_per_label.csv", index_col=0)

# Prefer best metrics from full json (handles incomplete CSV)
configs = [c["name"] for c in full]
best_auc = [c["best_macro_auc"] for c in full]
best_ap = [c["best_macro_ap"] for c in full]
times = [c["time_min"] for c in full]

# ---- 1. Macro AUC bar chart -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(configs, best_auc, color=colors, edgecolor="black", linewidth=0.8)
ax.axhline(0.5, ls="--", c="gray", alpha=0.6, label="Chance (0.50)")
ax.set_ylabel("Best Macro AUC")
ax.set_title("Modality Ablation — Best Macro AUC per Config")
ax.set_ylim(0, 1.05)
for bar, v in zip(bars, best_auc):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "macro_auc_bar.png", dpi=180)
plt.close()

# ---- 2. Per-label heatmap ---------------------------------------------------
matrix = per_label.fillna(np.nan).astype(float)
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels(matrix.columns, rotation=20, ha="right")
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.55 < v < 0.95 else "white", fontsize=8)
fig.colorbar(im, ax=ax, label="AUC", shrink=0.7)
ax.set_title("Per-Label AUC Across Modality Configurations")
plt.tight_layout()
plt.savefig(PLOT_DIR / "per_label_heatmap.png", dpi=180)
plt.close()

# ---- 3. Training curves -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for c, color in zip(full, colors):
    epochs = [h["epoch"] for h in c["history"]]
    aucs = [h["macro_auc"] for h in c["history"]]
    ax.plot(epochs, aucs, marker="o", linewidth=2, label=c["name"], color=color)
ax.axhline(0.5, ls="--", c="gray", alpha=0.6, label="Chance")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Macro AUC")
ax.set_title("Training Curves — Post-Epoch-1 Collapse Visible for Text Configs")
ax.set_ylim(0.4, 1.05)
ax.legend(loc="center right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "training_curves.png", dpi=180)
plt.close()

print("Saved:")
for p in sorted(PLOT_DIR.glob("*.png")):
    print(" ", p)
