"""
Graph 1 — Main experimental results for Paper 2.
All values from results_log.csv (verified).
Shows 5 configurations across 4 metrics: Accuracy, Macro-F1, AUC, Kappa.
Adds Grade-1 sensitivity as a separate panel.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Verified values from results_log.csv ─────────────────────────────────────
# run_name             | acc    | f1     | auc    | kappa  | g1_sens
# vqc_only             | 0.5455 | 0.2369 | 0.6455 | 0.2885 | 0.0000
# vit_baseline         | 0.7836 | 0.6376 | 0.9363 | 0.8811 | 0.7321
# hybrid_vit_vqc       | 0.8164 | 0.6588 | 0.8938 | 0.8711 | 0.5536
# hybrid_g1boost       | 0.7455 | 0.5242 | 0.9014 | 0.7619 | 0.7857
# hybrid_best (gated)  | 0.8145 | 0.6492 | 0.8939 | 0.8720 | 0.6607

configs = [
    "VQC-only\n(Config B)",
    "ViT-only\n(Config A)",
    "ViT+VQC\n(Proposed)",
    "ViT+VQC\nG1-Boost",
    "ViT+VQC\nGated Save",
]

accuracy  = [54.55, 78.36, 81.64, 74.55, 81.45]
macro_f1  = [0.237, 0.638, 0.659, 0.524, 0.649]
auc       = [0.646, 0.936, 0.894, 0.901, 0.894]
kappa     = [0.289, 0.881, 0.871, 0.762, 0.872]
g1_sens   = [0.000, 73.21, 55.36, 78.57, 66.07]  # % for readability

x = np.arange(len(configs))
w = 0.18

# Colours: VQC-only = red, ViT-only = blue, proposed group = greens
bar_colors = ["#E74C3C", "#2980B9", "#27AE60", "#F39C12", "#8E44AD"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
fig.subplots_adjust(wspace=0.38)

# ── Panel A: Accuracy, Macro-F1, Kappa grouped bars ──────────────────────────
ax1 = axes[0]
metrics_A = [
    ("Accuracy (%)",  accuracy,  "#4C72B0"),
    ("Macro-F1",      [v*100 for v in macro_f1],  "#DD8452"),
    ("Kappa",         [v*100 for v in kappa],     "#55A868"),
]
offsets = [-w, 0, w]
for (label, vals, col), off in zip(metrics_A, offsets):
    bars = ax1.bar(x + off, vals, width=w, color=col, edgecolor="white",
                   linewidth=0.6, label=label, zorder=3)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f"{v:.1f}", ha="center", va="bottom",
                 fontsize=6.5, rotation=90)

ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=8.5)
ax1.set_ylabel("Score (%, or ×100 for F1/Kappa)", fontsize=9)
ax1.set_ylim(0, 105)
ax1.set_yticks(range(0, 105, 10))
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
ax1.set_title("(a) Accuracy, Macro-F1, and Kappa", fontsize=9.5, pad=8)

# Highlight proposed model
ax1.axvspan(1.65, 2.35, alpha=0.06, color="#27AE60", zorder=0)

# ── Panel B: Grade-1 sensitivity (single metric, most clinically important) ──
ax2 = axes[1]
b2 = ax2.bar(x, g1_sens, width=0.55, color=bar_colors,
             edgecolor="white", linewidth=0.8, zorder=3)
for bar, v in zip(b2, g1_sens):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.8,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=8.5)
ax2.set_ylabel("Grade-1 Sensitivity (%)", fontsize=9)
ax2.set_ylim(0, 95)
ax2.set_yticks(range(0, 95, 10))
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("(b) Grade-1 (Early DR) Sensitivity", fontsize=9.5, pad=8)
ax2.axhline(y=0, color="#E74C3C", linewidth=0.8, linestyle=":", alpha=0.5)

# Legend for panel B
legend_entries = [mpatches.Patch(color=c, label=l) for c, l in zip(
    bar_colors, ["VQC-only", "ViT-only", "ViT+VQC (Proposed)", "G1-Boost", "Gated Save"])]
ax2.legend(handles=legend_entries, fontsize=7.5, loc="upper right",
           framealpha=0.9, edgecolor="#CCCCCC")

fig.suptitle("Fig. X. Main experimental results on the APTOS 2019 held-out test set (n = 550)",
             fontsize=10, y=1.01)

out = Path("paper/fig_main_results_graph.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/fig_main_results_graph.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.close(fig)

# ── Print verification table ──────────────────────────────────────────────────
print("\nVERIFICATION — exact values used in Graph 1:")
print(f"{'Config':<22} {'Acc%':>6} {'F1':>6} {'AUC':>6} {'Kappa':>6} {'G1%':>6}")
print("-"*56)
rows = list(zip(configs, accuracy, macro_f1, auc, kappa, g1_sens))
for r in rows:
    print(f"{r[0].replace(chr(10),' '):<22} {r[1]:>6.2f} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>6.3f} {r[5]:>6.2f}")
