"""Block diagram of the ViT+VQC architecture and dual-XAI pipeline for the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(9, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(-0.6, 7.6)
ax.axis("off")

def box(x, y, w, h, text, fc="#eef2ff", ec="#3b3b8f", fontsize=9, style="round,pad=0.08"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.3, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, zorder=3, wrap=True)
    return (x, y, w, h)

def arrow(b1, b2, text=None, color="#333333", style="-|>", connectionstyle="arc3,rad=0.0"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = (x1 + w1, y1 + h1/2)
    p2 = (x2, y2 + h2/2)
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=color,
                         lw=1.3, zorder=1, connectionstyle=connectionstyle)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.18
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=7.5, color=color)

# Main forward pipeline (left to right, middle row)
b_input  = box(0.1, 2.6, 1.5, 1.0, "Fundus\nImage\n224x224", fc="#f5f5f5")
b_vit    = box(1.9, 2.4, 1.9, 1.4, "ViT-Base/16\nBackbone\n(frozen, 86.4M)", fc="#dbe7ff")
b_reduce = box(4.1, 2.6, 1.5, 1.0, "Linear\n768 -> 4", fc="#e2f0e2")
b_vqc    = box(5.9, 2.2, 2.0, 1.8, "4-Qubit VQC\n2 variational layers\n(PennyLane,\nparameter-shift)", fc="#fde8d2")
b_head   = box(8.2, 2.6, 1.5, 1.0, "Linear\n4 -> 5 / 4 -> 1", fc="#e2f0e2")
b_out    = box(10.0, 2.5, 1.7, 1.2, "DR Grade (5-cls)\nor DME (binary)", fc="#f5f5f5")

arrow(b_input, b_vit)
arrow(b_vit, b_reduce, text="[CLS] 768-d")
arrow(b_reduce, b_vqc, text="angle encode")
arrow(b_vqc, b_head, text="<Z1..Z4>")
arrow(b_head, b_out)

# Grad-CAM branch (from ViT, going down)
b_gradcam = box(1.6, 0.2, 2.5, 1.2, "Grad-CAM\n(layernorm_before,\nlast encoder layer)\n+ fundus mask", fc="#ffe0e0")
gx1, gy1, gw1, gh1 = b_vit
gx2, gy2, gw2, gh2 = b_gradcam
a = FancyArrowPatch((gx1 + gw1/2, gy1), (gx2 + gw2/2, gy2 + gh2), arrowstyle="-|>",
                     mutation_scale=14, color="#8a2f2f", lw=1.2,
                     connectionstyle="arc3,rad=0.15")
ax.add_patch(a)
ax.text(gx1 + gw1/2 - 0.9, (gy1+gy2+gh2)/2 - 0.15, "gradients\n(WHERE)", fontsize=7.5, color="#8a2f2f")

# TCAV branch (from CLS features, going up)
b_tcav = box(4.0, 5.0, 2.6, 1.3, "TCAV\n(3 concept probes on\n[CLS] features, live\nper-image scoring)", fc="#e0e8ff")
tx1, ty1, tw1, th1 = b_reduce
tx2, ty2, tw2, th2 = b_tcav
a = FancyArrowPatch((tx1 + tw1/2, ty1 + th1), (tx2 + tw2/2, ty2), arrowstyle="-|>",
                     mutation_scale=14, color="#2f4a8a", lw=1.2,
                     connectionstyle="arc3,rad=-0.15")
ax.add_patch(a)
ax.text(tx1 + tw1/2 - 0.3, (ty1+th1+ty2)/2 + 0.05, "[CLS] features\n(WHY)", fontsize=7.5, color="#2f4a8a")

# Concept patch source note
ax.text(6.05, 7.15, "Concept patches cropped from IDRiD pixel-level lesion masks\n"
                     "(hemorrhage = microaneurysm + haemorrhage blobs; exudate = hard + soft exudate blobs;\n"
                     "edema = exudate blobs near macula center)",
        fontsize=7, ha="center", va="top", color="#2f4a8a")

ax.text(2.85, -0.5, "Faithfulness validated against the same IDRiD pixel-level masks (IoU / recall / precision)",
        fontsize=7, ha="center", va="bottom", color="#8a2f2f")

fig.suptitle("QuantumRetina-X: Hybrid ViT+VQC Architecture with Dual Explainability", fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig("paper/fig_architecture.png", dpi=200, bbox_inches="tight")
print("saved paper/fig_architecture.png")
