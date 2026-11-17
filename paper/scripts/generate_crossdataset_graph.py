"""
Graph 2 — External / Cross-Dataset Evaluation for Paper 2.
All values from cross_dataset_summary.csv (verified).
Grouped bar chart: ViT-only vs ViT+VQC across 3 external datasets.
Two panels: (a) Accuracy  (b) Kappa — both directly comparable across datasets.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Verified values from cross_dataset_summary.csv ───────────────────────────
datasets   = ["IDRiD\n(India, n=455)", "Messidor\n(France, n=1,744)", "EyePACS\n(USA, n=35,126)"]

vit_acc    = [42.86, 59.40, 63.63]
vqc_acc    = [43.52, 58.31, 62.86]

vit_kappa  = [0.5781, 0.4492, 0.4068]
vqc_kappa  = [0.6184, 0.4302, 0.4056]

vit_g1     = [31.82, 3.33,  16.99]
vqc_g1     = [50.00, 5.56,  18.13]

x = np.arange(len(datasets))
w = 0.35

C_VIT = "#2980B9"
C_VQC = "#27AE60"

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")
fig.subplots_adjust(wspace=0.42)

# ── Panel A: Accuracy ─────────────────────────────────────────────────────────
ax1 = axes[0]
b1a = ax1.bar(x - w/2, vit_acc, w, color=C_VIT, edgecolor="white", label="ViT-only", zorder=3)
b1b = ax1.bar(x + w/2, vqc_acc, w, color=C_VQC, edgecolor="white", label="ViT+VQC (ours)", zorder=3)
for bars in [b1a, b1b]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(datasets, fontsize=8.5)
ax1.set_ylabel("Accuracy (%)", fontsize=9)
ax1.set_ylim(0, 85); ax1.set_yticks(range(0, 86, 10))
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")
ax1.set_title("(a) Accuracy", fontsize=9.5, pad=8)

# ── Panel B: Kappa ────────────────────────────────────────────────────────────
ax2 = axes[1]
b2a = ax2.bar(x - w/2, vit_kappa, w, color=C_VIT, edgecolor="white", label="ViT-only", zorder=3)
b2b = ax2.bar(x + w/2, vqc_kappa, w, color=C_VQC, edgecolor="white", label="ViT+VQC (ours)", zorder=3)
for bars in [b2a, b2b]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(datasets, fontsize=8.5)
ax2.set_ylabel("Quadratic-Weighted Kappa", fontsize=9)
ax2.set_ylim(0, 0.85); ax2.set_yticks([i/10 for i in range(0, 9)])
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_title("(b) Quadratic-Weighted Kappa", fontsize=9.5, pad=8)

# ── Panel C: Grade-1 Sensitivity ─────────────────────────────────────────────
ax3 = axes[2]
b3a = ax3.bar(x - w/2, vit_g1, w, color=C_VIT, edgecolor="white", label="ViT-only", zorder=3)
b3b = ax3.bar(x + w/2, vqc_g1, w, color=C_VQC, edgecolor="white", label="ViT+VQC (ours)", zorder=3)
for bars in [b3a, b3b]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
ax3.set_xticks(x); ax3.set_xticklabels(datasets, fontsize=8.5)
ax3.set_ylabel("Grade-1 Sensitivity (%)", fontsize=9)
ax3.set_ylim(0, 65); ax3.set_yticks(range(0, 66, 10))
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
ax3.set_title("(c) Grade-1 (Early DR) Sensitivity", fontsize=9.5, pad=8)

fig.suptitle(
    "Fig. Y. Zero-shot cross-dataset evaluation: ViT-only vs ViT+VQC, trained on APTOS (India) only",
    fontsize=10, y=1.02)

out = Path("paper/fig_crossdataset_graph.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/fig_crossdataset_graph.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.close(fig)

print("\nVERIFICATION — exact values used:")
print(f"{'Dataset':<30} {'ViT-Acc':>8} {'VQC-Acc':>8} {'ViT-K':>7} {'VQC-K':>7} {'ViT-G1':>7} {'VQC-G1':>7}")
print("-"*74)
for i, ds in enumerate(["IDRiD (n=455)", "Messidor (n=1744)", "EyePACS (n=35126)"]):
    print(f"{ds:<30} {vit_acc[i]:>8.2f} {vqc_acc[i]:>8.2f} {vit_kappa[i]:>7.4f} {vqc_kappa[i]:>7.4f} {vit_g1[i]:>7.2f} {vqc_g1[i]:>7.2f}")
