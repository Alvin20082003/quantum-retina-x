"""
Generate publication-quality comparison bar graph for Paper 2 Table V.
Only five-class DR grading studies on APTOS 2019 are plotted for accuracy comparison.
Binary studies are listed in the table but excluded from the accuracy graph.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Verified data: FIVE-CLASS grading on APTOS 2019 only ──────────────────────
# These are the only studies that are directly comparable on accuracy.
# Source for each value is noted.

methods = [
    # (label, accuracy, category, reference)
    ("FQ-Conv-ViT\n(Springer 2025)",         87.97, "transformer",  "FQ-Conv-ViT, Springer SIViP 2025"),
    ("D-TNet\n(Sci.Rep. 2025)",              97.00, "cnn-trans",    "D-TNet, Sci.Rep. 2025"),
    ("TOViT\n(Sci.Rep. 2025)",               99.00, "transformer",  "TOViT, Sci.Rep. 2025 — note: multiple datasets"),
    ("CNN-Trans Ensemble\n(Sci.Rep. 2026)",  93.10, "cnn-trans",    "CNN-Trans ensemble, Sci.Rep. 2026"),
    ("ViT-only\n(ours)",                     78.40, "ours",         "This work — APTOS 5-class, frozen ViT"),
    ("ViT+VQC\n(ours)",                      81.60, "ours",         "This work — APTOS 5-class, ViT+4-qubit VQC"),
]

colors = {
    "transformer": "#2980B9",
    "cnn-trans":   "#8E44AD",
    "quantum":     "#E67E22",
    "ours":        "#27AE60",
}
hatch = {
    "transformer": "",
    "cnn-trans":   "//",
    "quantum":     "xx",
    "ours":        "",
}

labels  = [m[0] for m in methods]
accs    = [m[1] for m in methods]
cats    = [m[2] for m in methods]
bar_colors  = [colors[c] for c in cats]
bar_hatches = [hatch[c] for c in cats]

fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
x = np.arange(len(labels))
bars = ax.bar(x, accs, width=0.58,
              color=bar_colors, edgecolor="white", linewidth=0.8,
              zorder=3)

# Apply hatch separately for accessibility
for bar, h in zip(bars, bar_hatches):
    bar.set_hatch(h)

# Value labels above bars
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color="#1A1A2E")

# Reference line at our ViT+VQC
ax.axhline(y=81.6, color="#27AE60", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_ylim(60, 105)
ax.set_yticks(range(60, 106, 5))
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_entries = [
    mpatches.Patch(facecolor=colors["transformer"], label="Transformer-based"),
    mpatches.Patch(facecolor=colors["cnn-trans"],   hatch="//", label="CNN–Transformer hybrid"),
    mpatches.Patch(facecolor=colors["ours"],        label="Proposed (ours)"),
]
ax.legend(handles=legend_entries, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#CCCCCC")

ax.set_title("Accuracy comparison on five-class DR grading (APTOS 2019)",
             fontsize=10, pad=10)

# Footnote
fig.text(0.5, -0.04,
         "* TOViT result averaged across multiple datasets; D-TNet and CNN-Trans use full end-to-end fine-tuning.\n"
         "  Ours uses only 3,125 trainable parameters (frozen ViT backbone). Binary-task results excluded.",
         ha="center", fontsize=7.5, color="#555555", wrap=True)

fig.tight_layout()
out = Path("paper/fig_comparison_graph.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("paper/fig_comparison_graph.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.close(fig)
