"""
Final Paper 2 output: 2 clean tables + 2 publication-quality graphs.

TABLE 1: Our final results (ablation only - 3 rows, 4 metrics)
TABLE 2: Existing model comparison (DME + explainability focused)

GRAPH 1: Our results - accuracy+kappa+AUC+G1-sens for 3 configs
GRAPH 2: Existing model DME comparison
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

Path("paper").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — OUR FINAL RESULTS (clean, 3 configs only)
# ─────────────────────────────────────────────────────────────────────────────
# VERIFIED from results_log.csv — only the 3 ablation configs
configs = ["VQC-only\n(no ViT backbone)", "ViT-only\n(classical baseline)", "ViT+VQC\n(proposed)"]
acc    = [54.55, 78.36, 81.64]
auc    = [0.646,  0.936,  0.894]
kappa  = [0.289,  0.881,  0.871]
g1     = [0.000,  73.21,  55.36]

x  = np.arange(3)
w  = 0.20
C  = ["#E74C3C", "#2980B9", "#27AE60"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
fig.subplots_adjust(wspace=0.40)

# Panel A — Accuracy, AUC×100, Kappa×100
ax = axes[0]
offsets = [-w, 0, w]
metrics = [
    ("Accuracy (%)",    acc,                      "#4C72B0"),
    ("AUC × 100",       [v*100 for v in auc],     "#C44E52"),
    ("Kappa × 100",     [v*100 for v in kappa],   "#55A868"),
]
for (lbl, vals, col), off in zip(metrics, offsets):
    bars = ax.bar(x + off, vals, w, color=col, edgecolor="white",
                  label=lbl, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Score (% or ×100)", fontsize=9)
ax.set_ylim(0, 107)
ax.set_yticks(range(0, 105, 10))
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax.set_title("(a)  Accuracy, AUC, and Kappa", fontsize=10, pad=8)
# Highlight proposed model
ax.axvspan(1.6, 2.55, alpha=0.07, color="#27AE60", zorder=0)

# Panel B — Grade-1 Sensitivity only
ax2 = axes[1]
b2 = ax2.bar(x, g1, 0.52, color=C, edgecolor="white", linewidth=0.8, zorder=3)
for bar, v in zip(b2, g1):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=9)
ax2.set_ylabel("Grade-1 (Early DR) Sensitivity (%)", fontsize=9)
ax2.set_ylim(0, 90)
ax2.set_yticks(range(0, 91, 10))
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title("(b)  Grade-1 Early-Stage Sensitivity", fontsize=10, pad=8)
leg = [mpatches.Patch(color=c, label=l) for c, l in zip(C, configs)]
ax2.legend(handles=leg, fontsize=8.5, loc="upper right", framealpha=0.9)

fig.suptitle(
    "Fig. I.  Ablation study on APTOS 2019 held-out test set (n = 550)",
    fontsize=10.5, y=1.01
)
fig.savefig("paper/final_graph1_our_results.png",
            dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/final_graph1_our_results.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph 1 saved → paper/final_graph1_our_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — EXISTING MODEL COMPARISON (DME, same task, fair comparison)
# DMENet source: PLoS ONE 2020  — Acc=96.12%, Sens=96.32%, Spec=95.84%
# Bressler 2025 source: Ophthalmology Science — AUC=0.954, Sens=88.9%, Spec=88.9%
# Ours (results_log.csv): Acc=89.1%, AUC=0.928, Sens=78.5%, Spec=91.7%
# ─────────────────────────────────────────────────────────────────────────────
methods = ["DMENet\n[PLoS ONE 2020]", "Bressler et al.\n[Ophthal.Sci. 2025]", "ViT+VQC-DME\n(proposed)"]
sens = [96.32, 88.9,  78.50]
spec = [95.84, 88.9,  91.70]
aucs = [None,  95.4,  92.78]   # DMENet does not report AUC
MC   = ["#9B59B6", "#3498DB", "#27AE60"]

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
fig2.subplots_adjust(wspace=0.45)

# Panel A — Sensitivity + Specificity
xa = np.arange(3); wa = 0.32
ax3 = axes2[0]
b3a = ax3.bar(xa - wa/2, sens, wa, color=MC, edgecolor="white",
              label=None, zorder=3)
b3b = ax3.bar(xa + wa/2, spec, wa, color=[c+"77" for c in ["#9B59B6","#3498DB","#27AE60"]],
              edgecolor="white", linewidth=1.2, zorder=3)
for xi, (s, sp) in enumerate(zip(sens, spec)):
    ax3.text(xi - wa/2, s + 0.5, f"{s:.1f}%", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")
    ax3.text(xi + wa/2, sp + 0.5, f"{sp:.1f}%", ha="center", va="bottom",
             fontsize=8.5)
ax3.set_xticks(xa)
ax3.set_xticklabels(methods, fontsize=8.5)
ax3.set_ylabel("(%)", fontsize=9)
ax3.set_ylim(0, 110)
ax3.set_yticks(range(0, 110, 10))
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
legend_handles = [
    mpatches.Patch(facecolor="grey",       label="Sensitivity (solid)"),
    mpatches.Patch(facecolor="grey", alpha=0.45, label="Specificity (light)"),
]
ax3.legend(handles=legend_handles, fontsize=8, framealpha=0.9)
ax3.set_title("(a)  Sensitivity and Specificity", fontsize=10, pad=8)

# Panel B — AUC (only 2 methods report it)
methods_auc = ["Bressler et al.\n[Ophthal.Sci. 2025]", "ViT+VQC-DME\n(proposed)"]
auc_vals    = [95.4, 92.78]
auc_colors  = ["#3498DB", "#27AE60"]
xb = np.arange(2)
ax4 = axes2[1]
b4 = ax4.bar(xb, auc_vals, 0.45, color=auc_colors, edgecolor="white", zorder=3)
for bar, v in zip(b4, auc_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold")
ax4.set_xticks(xb)
ax4.set_xticklabels(methods_auc, fontsize=8.5)
ax4.set_ylabel("AUC (%)", fontsize=9)
ax4.set_ylim(80, 100)
ax4.set_yticks(range(80, 101, 2))
ax4.grid(axis="y", alpha=0.3, zorder=0)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_title("(b)  AUC  (DMENet did not report AUC)", fontsize=10, pad=8)

fig2.suptitle(
    "Fig. II.  Comparison with existing DME detection methods\n"
    "All methods: binary DME detection from fundus images",
    fontsize=10.5, y=1.03
)
fig2.savefig("paper/final_graph2_existing_comparison.png",
             dpi=200, bbox_inches="tight", facecolor="white")
fig2.savefig("paper/final_graph2_existing_comparison.pdf",
             bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("Graph 2 saved → paper/final_graph2_existing_comparison.png")
print("\nDone. Open both files in paper/ folder.")
