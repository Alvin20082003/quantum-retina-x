"""
Publication-quality ViT+VQC architecture diagram for Paper 2.
Three-panel layout: Data Layer | Proposed Framework | Evaluation Layer
Integrates Paper 1 visuals: fundus samples, patch grid, 768->4 compression, VQC measurement
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Arc
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from PIL import Image

# ─── Colors (IEEE academic palette) ─────────────────────────────────────────
C_BG       = "#FAFAFA"
C_LEFT     = "#EBF3FB"      # light blue – data layer
C_LEFT_BD  = "#2E75B6"
C_CTR      = "#EEF6EE"      # light green – model
C_CTR_BD   = "#2D7D32"
C_RIGHT    = "#FEF9E7"      # light yellow – evaluation
C_RIGHT_BD = "#B7950B"
C_VIT      = "#D6EAF8"
C_VIT_BD   = "#1A5276"
C_VQC      = "#D5F5E3"
C_VQC_BD   = "#1D6A39"
C_BOX      = "#FFFFFF"
C_ARROW    = "#2C3E50"
C_LABEL    = "#1A1A2E"
C_FREEZE   = "#E74C3C"
C_TRAIN    = "#27AE60"
C_GRADE    = ["#27AE60","#2980B9","#F39C12","#E67E22","#C0392B"]

FIG_W, FIG_H = 26, 15
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG, dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(C_BG)


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.15, zorder=2):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(b)
    return b


def label(ax, x, y, txt, fs=8, fw="normal", color=C_LABEL, ha="center", va="center", zorder=5):
    ax.text(x, y, txt, fontsize=fs, fontweight=fw, color=color,
            ha=ha, va=va, zorder=zorder,
            fontfamily="DejaVu Sans")


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, zorder=4,
          style="-|>", mut=12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=mut),
                zorder=zorder)


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL BORDERS
# ══════════════════════════════════════════════════════════════════════════════
LEFT_X, LEFT_W   = 0.15, 4.6
CTR_X,  CTR_W    = 5.1,  15.5
RIGHT_X, RIGHT_W = 20.9,  4.9
PY, PH = 0.4, 14.2

rbox(ax, LEFT_X,  PY, LEFT_W,  PH, C_LEFT,  C_LEFT_BD,  lw=2, radius=0.3, zorder=1)
rbox(ax, CTR_X,   PY, CTR_W,   PH, C_CTR,   C_CTR_BD,   lw=2.5, radius=0.3, zorder=1)
rbox(ax, RIGHT_X, PY, RIGHT_W, PH, C_RIGHT, C_RIGHT_BD, lw=2, radius=0.3, zorder=1)

# Panel titles
label(ax, LEFT_X+LEFT_W/2,   PY+PH-0.35, "Data Collection &\nPreprocessing Layer",
      fs=9, fw="bold", color=C_LEFT_BD)
label(ax, CTR_X+CTR_W/2,     PY+PH-0.35, "Proposed ViT–VQC Framework",
      fs=10, fw="bold", color=C_CTR_BD)
label(ax, RIGHT_X+RIGHT_W/2, PY+PH-0.35, "Evaluation &\nAnalysis Layer",
      fs=9, fw="bold", color=C_RIGHT_BD)


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — DATA & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset info box ──────────────────────────────────────────────────────────
rbox(ax, 0.3, 12.5, 4.2, 1.35, C_BOX, C_LEFT_BD, lw=1, radius=0.12)
label(ax, 2.4, 13.55, "APTOS 2019 Dataset", fs=8.5, fw="bold", color=C_LEFT_BD)
label(ax, 2.4, 13.15, "3,662 labeled fundus images", fs=7.5, color=C_LABEL)
label(ax, 2.4, 12.78, "Five DR grades  (0 – 4)", fs=7.5, color=C_LABEL)

# ── Fundus sample grid (5 grades, 1 row) from Fig 3 ─────────────────────────
grade_colors = ["#2ECC71","#3498DB","#F39C12","#E67E22","#C0392B"]
grade_labels_short = ["G-0\nHealthy","G-1\nMild","G-2\nMod.","G-3\nSev.","G-4\nPDR"]

# Try to load real images from data/archive
try:
    import pandas as pd
    root = Path("data/archive")
    df = pd.read_csv(root/"train.csv")
    img_index = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]}
    grade_imgs = {}
    for g in range(5):
        rows = df[df["diagnosis"]==g]
        for _, row in rows.iterrows():
            if row["id_code"] in img_index:
                try:
                    im = Image.open(img_index[row["id_code"]]).convert("RGB").resize((60,60))
                    grade_imgs[g] = np.array(im)
                    break
                except:
                    pass
except:
    grade_imgs = {}

cell_w, cell_h = 0.72, 0.72
gx0, gy0 = 0.35, 11.05
for i in range(5):
    cx = gx0 + i * (cell_w + 0.06)
    # border
    rbox(ax, cx, gy0, cell_w, cell_h, C_BOX, grade_colors[i], lw=1.5, radius=0.06)
    if i in grade_imgs:
        ax_img = ax.inset_axes([(cx+0.04)/FIG_W, (gy0+0.17)/FIG_H,
                                 (cell_w-0.08)/FIG_W, (cell_h-0.22)/FIG_H])
        ax_img.imshow(grade_imgs[i], aspect='auto')
        ax_img.axis("off")
    else:
        # placeholder circle
        c = Circle((cx+cell_w/2, gy0+cell_h/2+0.05), 0.22,
                    facecolor=grade_colors[i], alpha=0.3, zorder=3)
        ax.add_patch(c)
    label(ax, cx+cell_w/2, gy0+0.1, grade_labels_short[i],
          fs=6, fw="bold", color=grade_colors[i])

label(ax, 2.4, 10.78, "Representative APTOS Fundus Samples", fs=7, color=C_LABEL)

# ── Preprocessing steps ──────────────────────────────────────────────────────
prep_steps = [
    ("Resize", "224 × 224", "#AED6F1"),
    ("Normalize", "ImageNet\nμ/σ", "#A9DFBF"),
    ("Augment", "Flip·Rotate\nBrightness", "#FAD7A0"),
]
py_prep = 9.65
for i, (name, detail, col) in enumerate(prep_steps):
    px = 0.35 + i * 1.5
    rbox(ax, px, py_prep, 1.3, 0.85, col, C_LEFT_BD, lw=1, radius=0.1)
    label(ax, px+0.65, py_prep+0.57, name, fs=7.5, fw="bold", color=C_LABEL)
    label(ax, px+0.65, py_prep+0.25, detail, fs=6.5, color=C_LABEL)
    if i < 2:
        arrow(ax, px+1.3, py_prep+0.42, px+1.5, py_prep+0.42, lw=1.2)

label(ax, 2.4, 9.4, "Preprocessing Pipeline", fs=7, color=C_LABEL)

# ── Output fundus sample ──────────────────────────────────────────────────────
rbox(ax, 1.35, 8.1, 1.9, 1.05, C_BOX, C_LEFT_BD, lw=1.5, radius=0.1)
if 2 in grade_imgs:
    ax_main = ax.inset_axes([1.45/FIG_W, 8.2/FIG_H, 1.7/FIG_W, 0.85/FIG_H])
    ax_main.imshow(grade_imgs[2], aspect='auto')
    ax_main.axis("off")
label(ax, 2.3, 8.65, "Processed Fundus\nSample  224×224", fs=6.5, color=C_LEFT_BD, fw="bold")

# left→center main arrow
arrow(ax, 4.85, 10.0, 5.25, 10.0, lw=2, mut=14)
label(ax, 5.07, 10.22, "Input", fs=7, fw="bold", color=C_ARROW)


# ══════════════════════════════════════════════════════════════════════════════
#  CENTER PANEL — ViT + VQC FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

# ── Sub-panel borders ─────────────────────────────────────────────────────────
# ViT section
rbox(ax, 5.3, 1.1, 7.8, 12.5, C_VIT, C_VIT_BD, lw=1.5, radius=0.2, zorder=1)
label(ax, 9.2, 13.35, "A.  Vision Transformer Feature Extraction",
      fs=8.5, fw="bold", color=C_VIT_BD)

# VQC section
rbox(ax, 13.3, 1.1, 7.1, 12.5, C_VQC, C_VQC_BD, lw=1.5, radius=0.2, zorder=1)
label(ax, 16.85, 13.35, "B.  Variational Quantum Processing",
      fs=8.5, fw="bold", color=C_VQC_BD)


# ────────────────────────────────────────────────────────────────────────────
#  A1. Fundus input + Patch embedding (Fig 4 / Fig 7)
# ────────────────────────────────────────────────────────────────────────────
# Fundus input box
rbox(ax, 5.55, 10.9, 2.2, 1.65, C_BOX, C_VIT_BD, lw=1.2, radius=0.12)
if 2 in grade_imgs:
    ax_f = ax.inset_axes([5.65/FIG_W, 11.1/FIG_H, 1.8/FIG_W, 1.25/FIG_H])
    ax_f.imshow(grade_imgs[2], aspect='auto')
    ax_f.axis("off")
label(ax, 6.65, 10.98, "Fundus Input", fs=7, fw="bold", color=C_VIT_BD)

arrow(ax, 7.75, 11.72, 8.1, 11.72, lw=1.5)

# Patch grid box (Fig 7)
rbox(ax, 8.1, 10.9, 2.6, 1.65, C_BOX, C_VIT_BD, lw=1.2, radius=0.12)
# Draw patch grid visually
if 2 in grade_imgs:
    ax_pg = ax.inset_axes([8.2/FIG_W, 11.1/FIG_H, 1.5/FIG_W, 1.25/FIG_H])
    ax_pg.imshow(grade_imgs[2], aspect='auto', alpha=0.8)
    for gi in range(0, 15):
        ax_pg.axhline(gi/14, color='cyan', lw=0.4, alpha=0.8)
        ax_pg.axvline(gi/14, color='cyan', lw=0.4, alpha=0.8)
    ax_pg.axis("off")
label(ax, 9.6, 12.35, "Patch Embedding", fs=7, fw="bold", color=C_VIT_BD)
label(ax, 9.6, 12.12, "14×14 grid | 196 patches", fs=6.5, color=C_LABEL)
label(ax, 9.6, 11.88, "16×16 px each", fs=6.5, color=C_LABEL)
label(ax, 9.6, 10.98, "➝  197 tokens  (196 + [CLS])", fs=6.5, color=C_LABEL)

arrow(ax, 10.7, 11.72, 11.0, 11.72, lw=1.5)

# ────────────────────────────────────────────────────────────────────────────
#  A2. Frozen ViT-Base/16
# ────────────────────────────────────────────────────────────────────────────
rbox(ax, 11.0, 10.6, 1.85, 2.0, C_BOX, C_VIT_BD, lw=1.5, radius=0.12, zorder=3)
label(ax, 11.925, 12.38, "Frozen", fs=7, fw="bold", color=C_FREEZE)
label(ax, 11.925, 12.12, "ViT-Base/16", fs=8, fw="bold", color=C_VIT_BD)
label(ax, 11.925, 11.85, "12 Transformer", fs=6.5, color=C_LABEL)
label(ax, 11.925, 11.62, "Encoder Layers", fs=6.5, color=C_LABEL)
# Stacked blocks to represent transformer layers
for bi in range(4):
    by = 10.72 + bi * 0.22
    rbox(ax, 11.08, by, 1.7, 0.18, "#BDD7EE", C_VIT_BD, lw=0.5, radius=0.04, zorder=4)
# Lock symbol
label(ax, 12.78, 11.72, "[FROZEN]", fs=6.5, fw="bold", color=C_FREEZE)

arrow(ax, 12.85, 11.72, 12.97, 11.72, lw=1.5)

# ────────────────────────────────────────────────────────────────────────────
#  A3. CLS token representation
# ────────────────────────────────────────────────────────────────────────────
# Show 768-D token as colored strips
rbox(ax, 5.55, 8.9, 7.3, 1.6, C_BOX, C_VIT_BD, lw=1.2, radius=0.12, zorder=2)
np.random.seed(42)
strip_vals = np.random.randn(60)
xs = np.linspace(5.75, 12.55, 60)
for xi, v in zip(xs, strip_vals):
    col = "#E74C3C" if v > 0 else "#3498DB"
    ax.plot([xi, xi], [9.55, 9.85], color=col, lw=1.8, alpha=0.6, zorder=3)
label(ax, 9.2, 10.3, "768-D CLS Representation  [Global Retinal Feature Token]",
      fs=7.5, fw="bold", color=C_VIT_BD)
label(ax, 9.2, 9.42, "768 real feature values  ←  classification token from 12th encoder layer",
      fs=6.5, color=C_LABEL)
# Arrow from ViT to CLS
arrow(ax, 11.925, 10.6, 11.925, 10.5, lw=1.5)
# Down arrow from CLS box to next
arrow(ax, 9.2, 8.9, 9.2, 8.6, lw=1.5)

# ────────────────────────────────────────────────────────────────────────────
#  A4. Feature compression 768→4 (Fig 8 style)
# ────────────────────────────────────────────────────────────────────────────
rbox(ax, 5.55, 7.0, 7.3, 1.5, C_BOX, C_TRAIN, lw=1.5, radius=0.12, zorder=2)
label(ax, 9.2, 8.35, "Trainable Linear Compression   768 → 4   (W ∈ ℝ⁴ˣ⁷⁶⁸)",
      fs=7.5, fw="bold", color=C_TRAIN)
# Compressed 4-bar chart (Fig 8 style)
bar_vals = [0.110, -1.539, 1.418, -0.038]
bar_colors = ["#27AE60","#E74C3C","#27AE60","#E74C3C"]
bx0, bby, bw, bh = 6.2, 7.15, 0.42, 0.8
for bi, (bv, bc) in enumerate(zip(bar_vals, bar_colors)):
    bx = bx0 + bi * 0.55
    norm_h = abs(bv) / 2.0 * bh * 0.6
    by_base = bby + bh/2
    ax.bar(bx+0.21, bv/2.0 * bh * 0.6, width=0.35, bottom=by_base,
           color=bc, alpha=0.85, zorder=4)
    label(ax, bx+0.21, by_base - 0.12, f"x{bi}", fs=6, color=C_LABEL)
label(ax, 7.55, 7.52, f"{bar_vals[0]}  {bar_vals[1]}  {bar_vals[2]}  {bar_vals[3]}",
      fs=6.5, color=C_LABEL)

# big strip → 4 bars visual arrow
ax.annotate("", xy=(8.55, 7.75), xytext=(7.8, 7.75),
            arrowprops=dict(arrowstyle="-|>", color=C_TRAIN, lw=1.5, mutation_scale=10),
            zorder=4)

label(ax, 10.5, 7.85, "4 compressed\nfeature values", fs=7, fw="bold", color=C_TRAIN)
label(ax, 10.5, 7.35, "✓ Trainable layer", fs=6.5, color=C_TRAIN)

arrow(ax, 9.2, 7.0, 9.2, 6.8, lw=1.5)


# ════════════════════════════════════════════════════════════════════════════
#  B. VQC — Quantum Circuit (right sub-panel)
# ════════════════════════════════════════════════════════════════════════════

# Qubit lines
Q_X0, Q_X1 = 13.55, 19.8
qubit_ys = [12.1, 11.3, 10.5, 9.7]
qubit_labels = ["q₀","q₁","q₂","q₃"]
feat_labels  = ["x₀","x₁","x₂","x₃"]

for qi, (qy, ql, fl) in enumerate(zip(qubit_ys, qubit_labels, feat_labels)):
    # qubit wire
    ax.plot([Q_X0+0.5, Q_X1+0.1], [qy, qy], color="#555", lw=1.2, zorder=3)
    # input label
    rbox(ax, Q_X0-0.05, qy-0.22, 0.55, 0.44, "#F8F9FA", C_VQC_BD, lw=0.8, radius=0.06, zorder=5)
    label(ax, Q_X0+0.22, qy, f"|0⟩", fs=7, fw="bold", color=C_VQC_BD)
    label(ax, Q_X0-0.28, qy+0.05, fl, fs=7, fw="bold", color=C_LABEL)

# Feature input arrows from left
arrow(ax, 13.35, 6.8, 13.45, 10.9, lw=1.5, color=C_VQC_BD, style="-")
label(ax, 13.25, 8.85, "4-D\nInput", fs=6.5, fw="bold", color=C_VQC_BD)

# RY encoding gates
RY_X = Q_X0 + 0.7
for qi, qy in enumerate(qubit_ys):
    rbox(ax, RY_X-0.22, qy-0.22, 0.44, 0.44, "#FDEBD0", "#E67E22", lw=1, radius=0.06, zorder=5)
    label(ax, RY_X, qy, "RY", fs=6.5, fw="bold", color="#E67E22")
label(ax, RY_X, min(qubit_ys)-0.42, "Y-axis\nEncoding", fs=6, color="#E67E22")

# Layer 1 gate block
L1_X = RY_X + 0.9
rbox(ax, L1_X-0.05, qubit_ys[-1]-0.35, 1.65, qubit_ys[0]-qubit_ys[-1]+0.7,
     "#EBF5FB", C_VIT_BD, lw=1, radius=0.1, zorder=4)
label(ax, L1_X+0.77, qubit_ys[0]+0.28, "Layer 1", fs=6.5, fw="bold", color=C_VIT_BD)
for qi, qy in enumerate(qubit_ys):
    for gi, (gate, gc) in enumerate([("RX","#8E44AD"),("RY","#2980B9"),("RZ","#27AE60")]):
        gx = L1_X + 0.1 + gi * 0.46
        rbox(ax, gx-0.19, qy-0.19, 0.38, 0.38, C_BOX, gc, lw=0.8, radius=0.05, zorder=6)
        label(ax, gx, qy, gate, fs=5.5, fw="bold", color=gc)

# CNOT ring layer 1
CNOT_X = L1_X + 1.75
for qi in range(4):
    qy_ctrl = qubit_ys[qi]
    qy_tgt  = qubit_ys[(qi+1)%4]
    ax.plot([CNOT_X, CNOT_X], [min(qy_ctrl, qy_tgt), max(qy_ctrl, qy_tgt)],
            color="#555", lw=1.2, zorder=4)
    c_ctrl = Circle((CNOT_X, qy_ctrl), 0.1, facecolor="#555", zorder=5)
    ax.add_patch(c_ctrl)
    c_tgt = Circle((CNOT_X, qy_tgt), 0.18, facecolor="none", edgecolor="#555", lw=1.2, zorder=5)
    ax.add_patch(c_tgt)
    ax.plot([CNOT_X-0.18, CNOT_X+0.18], [qy_tgt, qy_tgt], color="#555", lw=1.2, zorder=5)
label(ax, CNOT_X, min(qubit_ys)-0.42, "CNOT\nRing", fs=6, color="#555")

# Layer 2
L2_X = CNOT_X + 0.45
rbox(ax, L2_X-0.05, qubit_ys[-1]-0.35, 1.65, qubit_ys[0]-qubit_ys[-1]+0.7,
     "#EBF5FB", C_VIT_BD, lw=1, radius=0.1, zorder=4)
label(ax, L2_X+0.77, qubit_ys[0]+0.28, "Layer 2", fs=6.5, fw="bold", color=C_VIT_BD)
for qi, qy in enumerate(qubit_ys):
    for gi, (gate, gc) in enumerate([("RX","#8E44AD"),("RY","#2980B9"),("RZ","#27AE60")]):
        gx = L2_X + 0.1 + gi * 0.46
        rbox(ax, gx-0.19, qy-0.19, 0.38, 0.38, C_BOX, gc, lw=0.8, radius=0.05, zorder=6)
        label(ax, gx, gate, fs=5.5, fw="bold", color=gc) if False else \
        label(ax, gx, qy, gate, fs=5.5, fw="bold", color=gc)

# CNOT ring layer 2
CNOT2_X = L2_X + 1.75
for qi in range(4):
    qy_ctrl = qubit_ys[qi]
    qy_tgt  = qubit_ys[(qi+1)%4]
    ax.plot([CNOT2_X, CNOT2_X], [min(qy_ctrl, qy_tgt), max(qy_ctrl, qy_tgt)],
            color="#555", lw=1.2, zorder=4)
    c2c = Circle((CNOT2_X, qy_ctrl), 0.1, facecolor="#555", zorder=5)
    ax.add_patch(c2c)
    c2t = Circle((CNOT2_X, qy_tgt), 0.18, facecolor="none", edgecolor="#555", lw=1.2, zorder=5)
    ax.add_patch(c2t)
    ax.plot([CNOT2_X-0.18, CNOT2_X+0.18], [qy_tgt, qy_tgt], color="#555", lw=1.2, zorder=5)
label(ax, CNOT2_X, min(qubit_ys)-0.42, "CNOT\nRing", fs=6, color="#555")

# Measurement boxes (Fig 9 style)
MEAS_X = CNOT2_X + 0.45
meas_vals = [-0.146, 0.097, -0.148, 0.097]
for qi, (qy, mv) in enumerate(zip(qubit_ys, meas_vals)):
    rbox(ax, MEAS_X-0.22, qy-0.22, 0.44, 0.44, "#FEF9E7", "#B7950B", lw=1, radius=0.06, zorder=5)
    label(ax, MEAS_X, qy, "Z_i", fs=6, fw="bold", color="#B7950B")
    label(ax, MEAS_X+0.33, qy, f"{mv:.3f}", fs=6, color=C_LABEL)

label(ax, MEAS_X, min(qubit_ys)-0.42, "Pauli-Z\nMeasure", fs=6, color="#B7950B")

# Sub-circuit labels
label(ax, (Q_X0+MEAS_X)/2, qubit_ys[0]+0.6,
      "4-qubit VQC  ·  2 variational layers  ·  RX / RY / RZ rotations  ·  CNOT-ring entanglement",
      fs=7, fw="bold", color=C_VQC_BD)

# ── Quantum feature vector ────────────────────────────────────────────────────
rbox(ax, 13.45, 7.8, 6.9, 1.45, C_BOX, C_VQC_BD, lw=1.5, radius=0.12, zorder=2)
bar2_vals = [-0.146, 0.097, -0.148, 0.097]
bx2_0 = 14.0
for bi, bv in enumerate(bar2_vals):
    bc = "#27AE60" if bv > 0 else "#E74C3C"
    ax.bar(bx2_0 + bi*0.55 + 0.25, bv*0.55, width=0.4, bottom=8.52,
           color=bc, alpha=0.85, zorder=4)
    label(ax, bx2_0 + bi*0.55 + 0.25, 8.0, f"q{bi}", fs=6.5, color=C_LABEL)
label(ax, 16.4, 9.08, "4-D Quantum Feature Vector  ⟨Z₀  Z₁  Z₂  Z₃⟩  ∈ [−1, +1]⁴",
      fs=7.5, fw="bold", color=C_VQC_BD)

arrow(ax, 16.9, 7.8, 16.9, 7.5, lw=1.5)

# ── Linear classifier ─────────────────────────────────────────────────────────
rbox(ax, 13.45, 6.3, 6.9, 1.1, "#D5EAD0", C_VQC_BD, lw=1.5, radius=0.12, zorder=2)
label(ax, 16.9, 7.15, "Trainable Linear Classifier    4 → 5    (W ∈ ℝ⁵ˣ⁴)", fs=7.5, fw="bold", color=C_VQC_BD)
label(ax, 16.9, 6.72, "Softmax  →  Class probabilities", fs=7, color=C_LABEL)

arrow(ax, 16.9, 6.3, 16.9, 5.9, lw=1.8, mut=14)

# ── Grade output nodes ────────────────────────────────────────────────────────
grade_full = ["Grade 0\nNo DR","Grade 1\nMild","Grade 2\nModerate","Grade 3\nSevere","Grade 4\nPDR"]
ox0 = 13.55
for gi, (gl, gc) in enumerate(zip(grade_full, C_GRADE)):
    gx = ox0 + gi * 1.42
    rbox(ax, gx, 4.75, 1.2, 0.95, gc+"33", gc, lw=1.5, radius=0.1, zorder=3)
    label(ax, gx+0.6, 5.22, gl, fs=6.5, fw="bold", color=gc)
    arrow(ax, 16.9, 5.9, gx+0.6, 5.7, lw=1.0, color=gc, mut=8)

label(ax, 16.9, 4.55, "Five-Class DR Prediction", fs=8, fw="bold", color=C_CTR_BD)

# Center→right arrow (from grades to evaluation)
arrow(ax, 20.7, 5.22, 21.1, 5.22, lw=2, mut=14)


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# ── A. Model comparison ──────────────────────────────────────────────────────
rbox(ax, 21.1, 11.7, 4.5, 2.0, C_BOX, C_RIGHT_BD, lw=1.2, radius=0.12)
label(ax, 23.35, 13.52, "A. Model Comparison", fs=7.5, fw="bold", color=C_RIGHT_BD)
configs = [("VQC-only","#E74C3C","54.6%"),
           ("ViT-only","#3498DB","78.4%"),
           ("ViT+VQC ★","#27AE60","81.6%")]
for ci, (cn, cc, acc) in enumerate(configs):
    cy = 12.85 - ci * 0.38
    rbox(ax, 21.25, cy-0.16, 2.5, 0.34, cc+"22", cc, lw=1, radius=0.06)
    label(ax, 21.75+0.75, cy+0.01, f"{cn}  {acc}", fs=7, fw="bold" if ci==2 else "normal", color=cc)

# ── B. Grade-1 sensitivity ───────────────────────────────────────────────────
rbox(ax, 21.1, 9.4, 4.5, 2.0, C_BOX, C_RIGHT_BD, lw=1.2, radius=0.12)
label(ax, 23.35, 11.22, "B. Grade-1 Sensitivity Study", fs=7.5, fw="bold", color=C_RIGHT_BD)
strats = [("Baseline Hybrid","#3498DB","55.4%"),
          ("G1-Boost ×3","#E67E22","78.6%"),
          ("Gated Save","#8E44AD","66.1%")]
for si, (sn, sc, sv) in enumerate(strats):
    sy = 10.55 - si * 0.38
    rbox(ax, 21.25, sy-0.16, 2.5, 0.34, sc+"22", sc, lw=1, radius=0.06)
    label(ax, 21.75+0.75, sy+0.01, f"{sn}  {sv}", fs=7, color=sc)

# ── C. External evaluation ───────────────────────────────────────────────────
rbox(ax, 21.1, 7.1, 4.5, 2.0, C_BOX, C_RIGHT_BD, lw=1.2, radius=0.12)
label(ax, 23.35, 8.92, "C. Zero-Shot External Evaluation", fs=7.5, fw="bold", color=C_RIGHT_BD)
datasets = [("Messidor (France)","#8E44AD","1,744 imgs","58.3%"),
            ("IDRiD (India)","#2980B9","455 imgs","43.5%"),
            ("EyePACS (USA)","#27AE60","35,126 imgs","62.9%")]
for di, (dn, dc, dn2, dacc) in enumerate(datasets):
    dy = 8.28 - di * 0.38
    rbox(ax, 21.25, dy-0.16, 4.2, 0.34, dc+"22", dc, lw=1, radius=0.06)
    label(ax, 21.25+2.1, dy+0.01, f"{dn}  ·  {dn2}  ·  Acc {dacc}", fs=6.5, color=dc)

# ── D. Metrics ───────────────────────────────────────────────────────────────
rbox(ax, 21.1, 1.2, 4.5, 5.6, C_BOX, C_RIGHT_BD, lw=1.2, radius=0.12)
label(ax, 23.35, 6.62, "D. Evaluation Metrics", fs=7.5, fw="bold", color=C_RIGHT_BD)
metrics = [("Accuracy",          "81.6%",  "#27AE60"),
           ("Macro-F1",           "0.659",  "#2980B9"),
           ("AUC-ROC",            "0.894",  "#8E44AD"),
           ("Q-W Kappa",          "0.871",  "#E67E22"),
           ("Grade-1 Sensitivity","55.4%",  "#E74C3C"),
           ("DME AUC",            "0.928",  "#16A085"),]
for mi, (mn, mv, mc) in enumerate(metrics):
    my = 6.0 - mi * 0.72
    rbox(ax, 21.25, my-0.26, 4.2, 0.52, mc+"18", mc, lw=1.2, radius=0.08)
    label(ax, 21.25+1.5, my, mn, fs=7, fw="bold", color=mc, ha="left")
    label(ax, 21.25+3.85, my, mv, fs=7.5, fw="bold", color=mc, ha="right")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL FLOW ARROWS (left ↔ center ↔ right)
# ══════════════════════════════════════════════════════════════════════════════

# Preprocessing → ViT input
arrow(ax, 4.78, 8.62, 5.45, 11.0, lw=1.8, color=C_ARROW, mut=12)

# VQC qubit wires connect to left side
arrow(ax, 13.35, 10.9, 13.55, 10.9, lw=1.5, color=C_VQC_BD)
arrow(ax, 13.35, 10.1, 13.55, 10.1, lw=1.5, color=C_VQC_BD)
arrow(ax, 13.35, 9.3, 13.55, 9.3, lw=1.5, color=C_VQC_BD)
arrow(ax, 13.35, 8.55, 13.55, 8.55, lw=1.5, color=C_VQC_BD)

# ViT output → VQC
arrow(ax, 12.9, 7.75, 13.35, 7.75, lw=1.8, color=C_CTR_BD, mut=12)

# Arrows from grade outputs to right panel
arrow(ax, 20.7, 12.7, 21.05, 12.7, lw=1.3, color=C_GRADE[0], mut=8)
arrow(ax, 20.7, 10.4, 21.05, 10.4, lw=1.3, color=C_GRADE[1], mut=8)
arrow(ax, 20.7, 8.1, 21.05, 8.1, lw=1.3, color=C_GRADE[2], mut=8)


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTION
# ══════════════════════════════════════════════════════════════════════════════
label(ax, FIG_W/2, 0.2,
      "Fig. 1.  Proposed ViT–VQC architecture.",
      fs=9, fw="bold", color=C_LABEL)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out = Path("paper/fig1_architecture.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
print(f"Saved {out}  ({out.stat().st_size//1024} KB)")

# Also save high-res PDF for IEEE submission
out_pdf = Path("paper/fig1_architecture.pdf")
fig.savefig(out_pdf, bbox_inches="tight", facecolor=C_BG)
print(f"Saved {out_pdf}")
plt.close(fig)
