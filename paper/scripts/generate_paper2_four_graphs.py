"""
Generate all 4 graphs for Paper 2.
All values from verified experiment records only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

Path("paper").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 1 — Main Experimental Results (DR classification ablation)
# Source: results_log.csv
# ─────────────────────────────────────────────────────────────────────────────
configs  = ["VQC-only", "ViT-only", "ViT+VQC\n(proposed)"]
acc      = [54.55, 78.36, 81.64]
macro_f1 = [0.237,  0.638,  0.659]
auc_val  = [0.646,  0.936,  0.894]
kappa    = [0.289,  0.881,  0.871]
g1       = [0.000, 73.21,  55.36]

x = np.arange(len(configs)); w = 0.19
C = ["#E74C3C","#2980B9","#27AE60"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
fig.subplots_adjust(wspace=0.38)

ax = axes[0]
for i, (lbl, vals, col) in enumerate([
        ("Accuracy (%)",  acc,                    "#4C72B0"),
        ("Macro-F1×100",  [v*100 for v in macro_f1], "#DD8452"),
        ("Kappa×100",     [v*100 for v in kappa],   "#55A868"),
        ("AUC×100",       [v*100 for v in auc_val], "#C44E52")]):
    bars = ax.bar(x + (i-1.5)*w, vals, w, color=col, edgecolor="white", label=lbl, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Score (%, or ×100 for unit metrics)", fontsize=9)
ax.set_ylim(0, 105); ax.set_yticks(range(0,105,10))
ax.grid(axis="y",alpha=0.3,zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax.set_title("(a) DR Classification — Four Metrics", fontsize=9.5, pad=8)
ax.axvspan(1.6,2.5,alpha=0.06,color="#27AE60",zorder=0)

ax2 = axes[1]
bars2 = ax2.bar(x, g1, 0.55, color=C, edgecolor="white", zorder=3)
for bar, v in zip(bars2, g1):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(configs, fontsize=9)
ax2.set_ylabel("Grade-1 Sensitivity (%)", fontsize=9)
ax2.set_ylim(0,90); ax2.set_yticks(range(0,91,10))
ax2.grid(axis="y",alpha=0.3,zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_title("(b) Grade-1 (Early DR) Sensitivity", fontsize=9.5, pad=8)
leg = [mpatches.Patch(color=c, label=l) for c,l in zip(C,configs)]
ax2.legend(handles=leg, fontsize=8, loc="upper right", framealpha=0.9)

fig.suptitle("Fig. I. Main experimental results — APTOS 2019 held-out test set (n=550)",
             fontsize=10, y=1.01)
fig.savefig("paper/p2_graph1_main_results.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/p2_graph1_main_results.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph 1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 2 — Cross-Dataset Evaluation
# Source: cross_dataset_summary.csv
# ─────────────────────────────────────────────────────────────────────────────
ds     = ["IDRiD\n(India, n=455)", "Messidor\n(France, n=1,744)", "EyePACS\n(USA, n=35,126)"]
v_acc  = [42.86, 59.40, 63.63]
q_acc  = [43.52, 58.31, 62.86]
v_k    = [0.578, 0.449, 0.407]
q_k    = [0.618, 0.430, 0.406]
v_g1   = [31.82, 3.33,  16.99]
q_g1   = [50.00, 5.56,  18.13]

x = np.arange(len(ds)); w=0.35
CV="#2980B9"; CQ="#27AE60"

fig, axes = plt.subplots(1, 3, figsize=(14,5), facecolor="white")
fig.subplots_adjust(wspace=0.42)

def dual_bar(ax, title, ylab, vvals, qvals, ylim, yticks):
    b1=ax.bar(x-w/2, vvals, w, color=CV, edgecolor="white", label="ViT-only", zorder=3)
    b2=ax.bar(x+w/2, qvals, w, color=CQ, edgecolor="white", label="ViT+VQC (ours)", zorder=3)
    for bars in [b1,b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ylim*0.012,
                    f"{bar.get_height():.1f}", ha="center",va="bottom",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=9); ax.set_ylim(0,ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y",alpha=0.3,zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=9.5, pad=8)

dual_bar(axes[0],"(a) Accuracy","Accuracy (%)",v_acc,q_acc,80,range(0,81,10))
dual_bar(axes[1],"(b) Quadratic-Weighted Kappa","Kappa",
         v_k, q_k, 0.80, [i/10 for i in range(0,9)])
dual_bar(axes[2],"(c) Grade-1 Sensitivity","Grade-1 Sensitivity (%)",
         v_g1, q_g1, 65, range(0,66,10))

fig.suptitle("Fig. II. Zero-shot cross-dataset evaluation — trained on APTOS only, no retraining",
             fontsize=10, y=1.02)
fig.savefig("paper/p2_graph2_crossdataset.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/p2_graph2_crossdataset.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph 2 saved")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 3 — XAI Results: TCAV scores by grade
# Source: tcav_scores.txt (verified)
# ─────────────────────────────────────────────────────────────────────────────
grades = ["G0\nHealthy", "G1\nMild", "G2\nModerate", "G3\nSevere", "G4\nPDR"]
tcav = {
    "Microaneurysm": [0.000, 0.000, 0.733, 1.000, 0.967],
    "Hemorrhage":    [0.000, 0.000, 0.600, 1.000, 0.900],
    "Exudate":       [0.000, 0.000, 0.500, 0.967, 0.833],
    "Edema":         [0.000, 0.000, 0.733, 1.000, 0.933],
}
colors_tcav = {"Microaneurysm":"#8E44AD","Hemorrhage":"#E24B4A",
               "Exudate":"#EF9F27","Edema":"#378ADD"}
x = np.arange(5); nk=len(tcav); bw=0.8/nk

fig, axes = plt.subplots(1, 2, figsize=(13,5), facecolor="white")
fig.subplots_adjust(wspace=0.40)

ax = axes[0]
for i,(concept,vals) in enumerate(tcav.items()):
    b=ax.bar(x+i*bw, vals, bw, label=concept, color=colors_tcav[concept], edgecolor="white", zorder=3)
ax.set_xticks(x+bw*(nk-1)/2); ax.set_xticklabels(grades, fontsize=9)
ax.set_ylabel("TCAV Score", fontsize=9)
ax.set_ylim(0,1.15); ax.set_yticks([i/10 for i in range(0,11)])
ax.axhline(0.5,color="gray",lw=1,ls="--",alpha=0.5,label="Chance baseline")
ax.grid(axis="y",alpha=0.3,zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=8, framealpha=0.9)
ax.set_title("(a) TCAV Concept Scores by DR Grade", fontsize=9.5, pad=8)

# Grad-CAM IoU per grade (from faithfulness_summary.txt)
gradcam_grades = ["G0\n(n=1)","G1\n(n=9)","G2\n(n=48)","G3\n(n=19)","G4\n(n=4)"]
iou_vals = [0.008, 0.002, 0.018, 0.093, 0.034]
rec_vals = [0.189, 0.059, 0.106, 0.216, 0.084]
pre_vals = [0.008, 0.002, 0.024, 0.156, 0.052]

ax2 = axes[1]
xg = np.arange(5); wg = 0.25
ax2.bar(xg-wg,   iou_vals, wg, color="#8b5cf6", edgecolor="white", label="IoU",       zorder=3)
ax2.bar(xg,      rec_vals, wg, color="#3b82f6", edgecolor="white", label="Recall",    zorder=3)
ax2.bar(xg+wg,   pre_vals, wg, color="#10b981", edgecolor="white", label="Precision", zorder=3)
ax2.set_xticks(xg); ax2.set_xticklabels(gradcam_grades, fontsize=8.5)
ax2.set_ylabel("Score", fontsize=9)
ax2.set_ylim(0,0.35); ax2.set_yticks([i/10 for i in range(0,4)])
ax2.grid(axis="y",alpha=0.3,zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=8, framealpha=0.9)
ax2.set_title("(b) Grad-CAM Faithfulness vs. Expert Masks (IDRiD, n=81)", fontsize=9.5, pad=8)

fig.suptitle("Fig. III. Explanation analysis results — TCAV concept scores and Grad-CAM faithfulness",
             fontsize=10, y=1.02)
fig.savefig("paper/p2_graph3_xai.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/p2_graph3_xai.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph 3 saved")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH 4 — Existing Related Models (Paper 2 specific: joint DR+DME + explainability)
# All external values from verified published sources
# ─────────────────────────────────────────────────────────────────────────────
# Only studies that (a) report a metric on a comparable task AND
# (b) are specifically relevant to Paper 2's joint DR+DME or explainability scope.
# DMENet [PLoS ONE 2020]: Acc=96.12%, Sens=96.32%, Spec=95.84% on Messidor+IDRiD (DME)
# Bressler et al. [5] [Ophthalmology Science 2025]: AUC=0.954, Sens=0.889, Spec=0.889 (DME)
# VGG16+Grad-CAM [17] [TVST 2020]: ECS=0.51 (best GradCAM faithfulness, closest comparable)
# CANet [11] [IEEE TMI 2020]: joint DR+DME on IDRiD/Messidor — best result on IDRiD
# Our DME detector: Acc=89.1%, AUC=0.928, Sens=78.5%, Spec=91.7%

# Graph: DME detection comparison (accuracy where available, else sensitivity)
# Only binary DME detection is used here — all methods do the same task
methods_dme = [
    "DMENet\n[PLoS ONE 2020]",
    "Bressler et al.\n[Ophthal.Sci.2025]",
    "ViT+VQC-DME\n(ours)",
]
dme_acc  = [96.12, None, 89.09]  # Bressler does not report accuracy
dme_sens = [96.32, 88.9,  78.50]
dme_spec = [95.84, 88.9,  91.70]
dme_auc  = [None,  95.4,  92.78]

fig, axes = plt.subplots(1, 2, figsize=(12,5), facecolor="white")
fig.subplots_adjust(wspace=0.40)

# Panel a: sensitivity + specificity
x4 = np.arange(len(methods_dme)); w4=0.32
ax = axes[0]
ax.bar(x4-w4/2, dme_sens, w4, color="#E74C3C", edgecolor="white", label="Sensitivity (%)", zorder=3)
ax.bar(x4+w4/2, dme_spec, w4, color="#3498DB", edgecolor="white", label="Specificity (%)", zorder=3)
for xi, (s,sp) in enumerate(zip(dme_sens, dme_spec)):
    ax.text(xi-w4/2, s+0.5, f"{s:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.text(xi+w4/2, sp+0.5,f"{sp:.1f}%",ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x4); ax.set_xticklabels(methods_dme, fontsize=8.5)
ax.set_ylabel("(%)", fontsize=9); ax.set_ylim(0,110); ax.set_yticks(range(0,111,10))
ax.grid(axis="y",alpha=0.3,zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.set_title("(a) DME Detection — Sensitivity and Specificity", fontsize=9.5, pad=8)

# Panel b: AUC (only for methods that report it)
methods_auc  = ["Bressler et al.\n[Ophthal.Sci.2025]", "ViT+VQC-DME\n(ours)"]
auc_plot = [95.4, 92.78]
colors_auc = ["#9B59B6","#27AE60"]
x4b = np.arange(len(methods_auc))
b4b = axes[1].bar(x4b, auc_plot, 0.45, color=colors_auc, edgecolor="white", zorder=3)
for bar, v in zip(b4b, auc_plot):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_xticks(x4b); axes[1].set_xticklabels(methods_auc, fontsize=8.5)
axes[1].set_ylabel("AUC (%)", fontsize=9); axes[1].set_ylim(80,100)
axes[1].set_yticks(range(80,101,2))
axes[1].grid(axis="y",alpha=0.3,zorder=0)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
axes[1].set_title("(b) DME Detection — AUC", fontsize=9.5, pad=8)

fig.suptitle("Fig. IV. Comparison with existing DME detection and joint DR/DME methods",
             fontsize=10, y=1.02)
fig.savefig("paper/p2_graph4_existing.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/p2_graph4_existing.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Graph 4 saved")
print("\nAll four Paper 2 graphs generated.")
