"""Real visual before/after for dimensionality reduction: the actual 768-number
[CLS] embedding (rendered as a value strip, the honest way to visualize a
non-spatial vector) next to the real 4-number reducer output, for the same
worked-example image already used throughout the paper."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils.config import load_config
from data.transforms import build_transforms
from models.hybrid import QuantumRetinaX

IMG_PATH = "data/idrid/Imagenes/Imagenes/IDRiD_001.jpg"

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = build_transforms(cfg, train=False)

model = QuantumRetinaX(cfg).to(device)
model.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True, map_location=device))
model.eval()

img_pil = Image.open(IMG_PATH).convert("RGB")
tensor = transform(img_pil).unsqueeze(0).to(device)

with torch.no_grad():
    cls = model.backbone(pixel_values=tensor).last_hidden_state[:, 0]
    reduced = model.reducer(cls)

cls_np = cls[0].cpu().numpy()
reduced_np = reduced[0].cpu().numpy()
print("CLS 768-dim: mean=%.3f std=%.3f norm=%.3f" % (cls_np.mean(), cls_np.std(), np.linalg.norm(cls_np)))
print("reduced 4-dim:", reduced_np)

fig, axes = plt.subplots(2, 1, figsize=(8, 3.2), gridspec_kw={"height_ratios": [1, 1.6]})

# Before: 768-dim CLS vector as a value strip (real values, not illustrative)
ax = axes[0]
im = ax.imshow(cls_np.reshape(1, -1), aspect="auto", cmap="RdBu_r", vmin=-np.abs(cls_np).max(), vmax=np.abs(cls_np).max())
ax.set_yticks([])
ax.set_xlabel("768 real [CLS] feature values (before reduction)", fontsize=9)
ax.set_title("Before: the real 768-number classification-token embedding", fontsize=10)

# After: 4-dim reduced vector as bars
ax2 = axes[1]
colors = ["#10b981" if v >= 0 else "#ef4444" for v in reduced_np]
ax2.bar([f"x{i}" for i in range(4)], reduced_np, color=colors)
ax2.axhline(0, color="#475569", linewidth=0.8)
ax2.set_title("After: the real 4-number reducer output (Equation 3)", fontsize=10)
for i, v in enumerate(reduced_np):
    ax2.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:.3f}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig("paper/fig_stage_reducer_visual.png", dpi=150)
print("saved paper/fig_stage_reducer_visual.png")
