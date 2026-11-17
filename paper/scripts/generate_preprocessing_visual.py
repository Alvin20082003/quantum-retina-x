"""Real visual before/after for preprocessing. The raw resized photo is shown
next to the actual normalized tensor the network receives, rendered as a real
per-channel value heatmap (not rescaled to look like an ordinary photo, since
that rescaling makes a linear transform look visually identical to the
original and hides the real change instead of showing it)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils.config import load_config
from data.transforms import build_transforms

IMG_PATH = "data/idrid/Imagenes/Imagenes/IDRiD_001.jpg"

cfg = load_config()
transform = build_transforms(cfg, train=False)

img_pil = Image.open(IMG_PATH).convert("RGB")
img_resized = img_pil.resize((224, 224))
tensor = transform(img_pil)  # C,H,W, normalized

norm_np = tensor.permute(1, 2, 0).numpy()  # H,W,C
print("normalized tensor: mean=%.3f std=%.3f min=%.3f max=%.3f" %
      (norm_np.mean(), norm_np.std(), norm_np.min(), norm_np.max()))

fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
axes[0].imshow(img_resized)
axes[0].set_title("Before: raw resized\nimage (0-255)", fontsize=10)
axes[0].axis("off")

chan_names = ["R", "G", "B"]
vmax = np.abs(norm_np).max()
for i, name in enumerate(chan_names):
    im = axes[i + 1].imshow(norm_np[:, :, i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[i + 1].set_title(f"After: real normalized\n{name} channel (Eq. 1)", fontsize=10)
    axes[i + 1].axis("off")
cbar = fig.colorbar(im, ax=axes[1:4], shrink=0.75, pad=0.02)
cbar.set_label("normalized value x'", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig.suptitle("Real preprocessing output for the example image: these are the actual numbers the network receives", fontsize=9.5)
fig.savefig("paper/fig_stage_preprocessing_visual.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_stage_preprocessing_visual.png")
