import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("experiments/results")
(out/"ablation").mkdir(parents=True, exist_ok=True)
(out/"cross_dataset").mkdir(parents=True, exist_ok=True)

models = ["VQC only\n(Config B)", "ViT only\n(Config A)", "ViT+VQC\n(Config C)"]
panels = [
    ("Accuracy",       [0.546, 0.784, 0.816], "#3498DB"),
    ("Kappa",          [0.289, 0.881, 0.871], "#2ECC71"),
    ("AUC-ROC",        [0.646, 0.936, 0.894], "#9B59B6"),
    ("G1-Sensitivity", [0.000, 0.732, 0.554], "#E74C3C"),
]
x = np.arange(3)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (title, vals, color) in zip(axes, panels):
    bars = ax.bar(x, vals, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_title(title); ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}",
                ha="center", fontsize=10)
fig.suptitle("Ablation Study: Component Contribution (single checkpoint per configuration)", fontsize=13)
fig.tight_layout()
fig.savefig(out/"ablation"/"ablation_chart.png", dpi=150)
plt.close(fig)
print("saved ablation_chart.png")

ds = ["APTOS\n(trained)", "Messidor\n(France)", "IDRiD\n(India)", "EyePACS\n(USA)"]
series = [
    ("Accuracy", [0.784,0.594,0.429,0.642], [0.816,0.583,0.435,0.630]),
    ("Kappa",    [0.881,0.449,0.578,0.412], [0.872,0.430,0.618,0.412]),
    ("AUC-ROC",  [0.936,0.760,0.786,0.724], [0.894,0.577,0.657,0.603]),
    ("G1-Sens",  [0.732,0.033,0.318,0.181], [0.554,0.056,0.500,0.190]),
]
x = np.arange(4); w = 0.35
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (title, vit, vqc) in zip(axes, series):
    ax.bar(x-w/2, vit, w, label="ViT-only", color="#4A90D9")
    ax.bar(x+w/2, vqc, w, label="ViT+VQC", color="#7F4FBF")
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=8)
    ax.set_title(title); ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Cross-Dataset Generalisation (trained on APTOS only, zero retraining)", fontsize=13)
fig.tight_layout()
fig.savefig(out/"cross_dataset"/"crossdataset_full_comparison.png", dpi=150)
plt.close(fig)
print("saved crossdataset_full_comparison.png")
