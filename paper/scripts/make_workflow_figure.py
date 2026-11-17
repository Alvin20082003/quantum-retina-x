"""Updated, complete end-to-end workflow flowchart: every real process stage
(preprocessing, fundus masking, the DME branch, and the red-free reference
view are now included, and the concept count is corrected from 3 to 4)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8.5))
ax.set_xlim(0, 15.5)
ax.set_ylim(-2.6, 9.4)
ax.axis("off")

def box(x, y, w, h, text, fc="#eef2ff", ec="#3b3b8f", fontsize=8.3, style="round,pad=0.08"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.3, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, zorder=3, wrap=True)
    return (x, y, w, h)

def arrow(b1, b2, text=None, color="#333333", fontsize=7.2, connectionstyle="arc3,rad=0.0", pos="mid"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = (x1 + w1, y1 + h1/2)
    p2 = (x2, y2 + h2/2)
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=color,
                         lw=1.2, zorder=1, connectionstyle=connectionstyle)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.18
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fontsize, color=color)

def vbox_arrow(b1, b2, text, color, connectionstyle):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    a = FancyArrowPatch((x1 + w1/2, y1 if y2 < y1 else y1 + h1), (x2 + w2/2, y2 + h2 if y2 < y1 else y2),
                         arrowstyle="-|>", mutation_scale=13, color=color, lw=1.2,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)
    ax.text(x1 + w1/2 + (0.9 if y2 < y1 else -0.9), (y1 + y2 + h2) / 2, text,
            fontsize=7.2, color=color, ha="center")

# ---- Main forward pass (middle row, y≈3.6-4.6) ----
b_input  = box(0.1, 3.7, 1.4, 1.0, "Fundus\nImage\n224x224", fc="#f5f5f5")
b_prep   = box(1.8, 3.7, 1.5, 1.0, "Preprocessing\nresize + normalize\n(Eq. 1)", fc="#fff6d6")
b_vit    = box(3.6, 3.5, 1.9, 1.4, "ViT-Base/16\nBackbone\n(frozen, 86.4M)", fc="#dbe7ff")
b_reduce = box(5.8, 3.7, 1.4, 1.0, "Linear\n768 -> 4\n(Eq. 3)", fc="#e2f0e2")
b_vqc    = box(7.5, 3.3, 1.9, 1.8, "4-Qubit VQC\n2 variational layers\n(Eqs. 4-6)", fc="#fde8d2")
b_head   = box(9.7, 3.7, 1.4, 1.0, "Linear\n4->5 / 4->1\n(Eqs. 7-8)", fc="#e2f0e2")
b_out    = box(11.4, 3.6, 1.6, 1.2, "DR Grade\n(5-class)", fc="#f5f5f5")

arrow(b_input, b_prep)
arrow(b_prep, b_vit)
arrow(b_vit, b_reduce, text="[CLS] 768-d")
arrow(b_reduce, b_vqc, text="angle encode")
arrow(b_vqc, b_head, text="<Z1..Z4>")
arrow(b_head, b_out)

# ---- DME branch (parallel model, below main row) ----
b_dme = box(9.7, 1.7, 3.3, 1.2, "2nd ViT+VQC model\n(same architecture,\nseparately trained, 1-output head)", fc="#e2f0e2")
b_dmeout = box(13.2, 1.75, 1.9, 1.1, "DME\nPresent / Absent", fc="#f5f5f5")
a = FancyArrowPatch((b_prep[0]+b_prep[2]/2, b_prep[1]), (b_dme[0]+0.3, b_dme[1]+b_dme[3]),
                     arrowstyle="-|>", mutation_scale=13, color="#1a7a3d", lw=1.2,
                     connectionstyle="arc3,rad=-0.25")
ax.add_patch(a)
ax.text(8.55, 2.75, "same preprocessed image", fontsize=7.2, color="#1a7a3d", ha="left")
arrow(b_dme, b_dmeout, color="#1a7a3d")

# ---- Grad-CAM branch (below ViT, red / WHERE) ----
b_mask = box(2.6, -0.6, 1.9, 1.1, "Fundus Mask\n(Otsu + ellipse,\n18% shrink)", fc="#ffe0e0")
b_gradcam = box(5.0, -0.6, 2.1, 1.1, "Grad-CAM backprop\n(layernorm_before,\nlast encoder layer)", fc="#ffe0e0")
b_heat = box(7.5, -0.6, 1.7, 1.1, "Heatmap\n(percentile-stretched,\nmasked)", fc="#ffe0e0")
vbox_arrow(b_vit, b_mask, "gradients\n(WHERE)", "#8a2f2f", "arc3,rad=0.2")
arrow(b_mask, b_gradcam, color="#8a2f2f")
arrow(b_gradcam, b_heat, color="#8a2f2f")

# ---- TCAV branch (above, blue / WHY) ----
b_score = box(3.4, 6.6, 2.3, 1.2, "Concept Scoring\n4 probes on [CLS]\n(microaneurysm, hemorrhage,\nexudate, edema)", fc="#e0e8ff")
b_maps = box(6.1, 6.6, 2.1, 1.2, "4 Per-Concept\nMaps (concept-CAM)", fc="#e0e8ff")
b_merge = box(8.6, 6.6, 2.1, 1.2, "Multi-Color Merge\n(winner-take-all,\nthreshold 0.3)", fc="#e0e8ff")
vbox_arrow(b_reduce, b_score, "[CLS] features\n(WHY)", "#2f4a8a", "arc3,rad=-0.25")
arrow(b_score, b_maps, color="#2f4a8a")
arrow(b_maps, b_merge, color="#2f4a8a")

# ---- Red-free branch (classical, gray, dashed) ----
b_redfree = box(0.1, 6.6, 1.9, 1.2, "Red-Free View\n(green channel + CLAHE)\nnot used by any model", fc="#eeeeee", ec="#888888")
a = FancyArrowPatch((b_input[0]+b_input[2]/2, b_input[1]+b_input[3]), (b_redfree[0]+b_redfree[2]/2, b_redfree[1]),
                     arrowstyle="-|>", mutation_scale=13, color="#888888", lw=1.1, linestyle="--",
                     connectionstyle="arc3,rad=0.2")
ax.add_patch(a)

ax.text(7.75, 8.35, "Concept patches cropped from real IDRiD pixel-level lesion masks", fontsize=7.5,
        ha="center", color="#2f4a8a")
ax.text(6.35, -1.4, "Faithfulness validated against the same IDRiD pixel-level masks (IoU / recall / precision)",
        fontsize=7.5, ha="center", color="#8a2f2f")

fig.suptitle("QuantumRetina-X: Complete End-to-End Process, Every Real Stage", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig("paper/fig_workflow_complete.png", dpi=200, bbox_inches="tight")
print("saved paper/fig_workflow_complete.png")
