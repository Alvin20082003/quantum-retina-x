"""
Generate 2 new graphs for Paper 1 additions:
1. Multi-seed error bar chart
2. Low-data comparison chart
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
Path("paper").mkdir(exist_ok=True)

# ── Graph A: Multi-seed error bars ────────────────────────────────────────────
# Verified from multiseed_results.json
vit_accs  = [0.7836, 0.8255, 0.8091]
hyb_accs  = [0.8164, 0.9109, 0.8982]
vit_kaps  = [0.8811, 0.8946, 0.8941]
hyb_kaps  = [0.8711, 0.9400, 0.9356]

vit_acc_mean, vit_acc_std = np.mean(vit_accs)*100, np.std(vit_accs)*100
hyb_acc_mean, hyb_acc_std = np.mean(hyb_accs)*100, np.std(hyb_accs)*100
vit_kap_mean, vit_kap_std = np.mean(vit_kaps),    np.std(vit_kaps)
hyb_kap_mean, hyb_kap_std = np.mean(hyb_kaps),    np.std(hyb_kaps)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
fig.subplots_adjust(wspace=0.38)

x = np.array([0, 1]); w = 0.35
C_VIT = "#2980B9"; C_HYB = "#27AE60"

# Panel A: Accuracy
ax = axes[0]
ax.bar(x[0]-w/2, vit_acc_mean, w, color=C_VIT, edgecolor="white", label="ViT-only", zorder=3)
ax.bar(x[0]+w/2, hyb_acc_mean, w, color=C_HYB, edgecolor="white", label="ViT+VQC", zorder=3)
ax.errorbar(x[0]-w/2, vit_acc_mean, yerr=vit_acc_std, fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
ax.errorbar(x[0]+w/2, hyb_acc_mean, yerr=hyb_acc_std, fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
# Per-seed scatter
for v in vit_accs: ax.scatter(x[0]-w/2, v*100, color="white", edgecolor=C_VIT, s=25, zorder=5)
for v in hyb_accs: ax.scatter(x[0]+w/2, v*100, color="white", edgecolor=C_HYB, s=25, zorder=5)
ax.text(x[0]-w/2, vit_acc_mean+vit_acc_std+0.5, f"{vit_acc_mean:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.text(x[0]+w/2, hyb_acc_mean+hyb_acc_std+0.5, f"{hyb_acc_mean:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks([]); ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.set_ylim(70, 98); ax.set_yticks(range(70, 99, 5))
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_title("(a)  Accuracy (mean ± std)", fontsize=9.5, pad=6)

# Panel B: Kappa
ax2 = axes[1]
ax2.bar(x[0]-w/2, vit_kap_mean, w, color=C_VIT, edgecolor="white", label="ViT-only", zorder=3)
ax2.bar(x[0]+w/2, hyb_kap_mean, w, color=C_HYB, edgecolor="white", label="ViT+VQC", zorder=3)
ax2.errorbar(x[0]-w/2, vit_kap_mean, yerr=vit_kap_std, fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
ax2.errorbar(x[0]+w/2, hyb_kap_mean, yerr=hyb_kap_std, fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
for v in vit_kaps: ax2.scatter(x[0]-w/2, v, color="white", edgecolor=C_VIT, s=25, zorder=5)
for v in hyb_kaps: ax2.scatter(x[0]+w/2, v, color="white", edgecolor=C_HYB, s=25, zorder=5)
ax2.text(x[0]-w/2, vit_kap_mean+vit_kap_std+0.003, f"{vit_kap_mean:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.text(x[0]+w/2, hyb_kap_mean+hyb_kap_std+0.003, f"{hyb_kap_mean:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks([]); ax2.set_ylabel("Quadratic-Weighted Kappa", fontsize=9)
ax2.set_ylim(0.82, 0.97); ax2.set_yticks([0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_title("(b)  Kappa (mean ± std)", fontsize=9.5, pad=6)

fig.suptitle("Fig. 5.  Multi-seed reproducibility (seeds 42, 123, 456). Dots show per-seed values.",
             fontsize=9, y=1.01)
fig.savefig("paper/fig_multiseed_errorbars.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph A saved")

# ── Graph B: Low-data comparison ──────────────────────────────────────────────
# Verified: full-data from results_log.csv, low-data from eval_lowdata run
configs = ["ViT-only\n(2,563)", "ViT+VQC\n(2,563)", "ViT-only\n(100)", "ViT+VQC\n(100)"]
accs    = [78.36, 81.64, 43.09, 59.82]
kaps    = [0.881, 0.871, 0.521, 0.761]
colors  = [C_VIT, C_HYB, C_VIT, C_HYB]
alphas  = [1.0, 1.0, 0.55, 0.55]

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")
fig2.subplots_adjust(wspace=0.38)

x4 = np.arange(4); w4=0.55
ax3 = axes2[0]
bars = ax3.bar(x4, accs, w4, color=colors, edgecolor="white", zorder=3)
for bar, a, v in zip(bars, alphas, accs):
    bar.set_alpha(a)
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax3.axvline(1.5, color="gray", lw=1.2, ls="--", alpha=0.6)
ax3.text(0.75, 73, "Full data", ha="center", fontsize=8, color="gray")
ax3.text(2.75, 55, "100 samples", ha="center", fontsize=8, color="gray")
# Annotation for quantum advantage
ax3.annotate("", xy=(3, 60.8), xytext=(2, 44.1),
             arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5))
ax3.text(2.5, 54, "+16.7%", fontsize=8.5, color="#E74C3C", fontweight="bold")
ax3.annotate("", xy=(1, 82.6), xytext=(0, 79.4),
             arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5))
ax3.text(0.5, 83, "+3.2%", fontsize=8.5, color="#E74C3C", fontweight="bold")
ax3.set_xticks(x4); ax3.set_xticklabels(configs, fontsize=8.5)
ax3.set_ylabel("Accuracy (%)", fontsize=9); ax3.set_ylim(0, 92)
ax3.set_yticks(range(0, 91, 10))
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
ax3.set_title("(a)  Accuracy — Full vs. Low-Data", fontsize=9.5, pad=6)
leg = [mpatches.Patch(color=C_VIT, label="ViT-only"),
       mpatches.Patch(color=C_HYB, label="ViT+VQC")]
ax3.legend(handles=leg, fontsize=8.5, framealpha=0.9)

ax4 = axes2[1]
bars2 = ax4.bar(x4, kaps, w4, color=colors, edgecolor="white", zorder=3)
for bar, a, v in zip(bars2, alphas, kaps):
    bar.set_alpha(a)
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax4.axvline(1.5, color="gray", lw=1.2, ls="--", alpha=0.6)
ax4.set_xticks(x4); ax4.set_xticklabels(configs, fontsize=8.5)
ax4.set_ylabel("Quadratic-Weighted Kappa", fontsize=9); ax4.set_ylim(0, 1.0)
ax4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax4.grid(axis="y", alpha=0.3, zorder=0)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
ax4.set_title("(b)  Kappa — Full vs. Low-Data", fontsize=9.5, pad=6)

fig2.suptitle("Fig. 6.  Low-data regime (100 training samples). Faded bars = 100-sample setting.",
              fontsize=9, y=1.01)
fig2.savefig("paper/fig_lowdata_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("Graph B saved")
print("Both graphs done.")
