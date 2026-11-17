"""Real, model/data-generated comparison images for Paper 2, Sections E and F.
Section E: a real DME-positive and a real DME-negative test image, both
correctly classified by the real trained model. Section F: real example
lesion patches actually used to train each TCAV concept probe, cropped
directly from the real IDRiD segmentation masks - no synthetic content."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_dme_loaders
from data.transforms import build_transforms
from models.hybrid import QuantumRetinaX
from xai.tcav_demo import extract_idrid_concept_patches, IDRID_COLORS, _blob_centroids

cfg = load_config(); set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = build_transforms(cfg, train=False)

# ============================================================
# SECTION E: real DME-positive and DME-negative correctly
# classified test images
# ============================================================
print("Loading real DME test set...")
_, _, dme_test_loader, dme_info = build_dme_loaders(cfg, num_workers=0)
dme_test_df = dme_test_loader.dataset.df
dme_img_index = dme_test_loader.dataset.img_index
print(f"Real held-out DME test set: {len(dme_test_df)} images")

dme_model = QuantumRetinaX(cfg, num_classes=1).to(device)
dme_model.load_state_dict(torch.load("weights/dme_vqc.pth", weights_only=True, map_location=device))
dme_model.eval()

found_pos, found_neg = None, None
with torch.no_grad():
    for _, row in dme_test_df.iterrows():
        true_dme = int(row["dme"])
        img_path = dme_img_index[str(row["id_code"])]
        img_pil = Image.open(img_path).convert("RGB")
        tensor = transform(img_pil).unsqueeze(0).to(device)
        prob = torch.sigmoid(dme_model(tensor)).item()
        pred = prob >= 0.5
        if true_dme == 1 and pred and found_pos is None:
            found_pos = (img_path, prob)
        if true_dme == 0 and not pred and found_neg is None:
            found_neg = (img_path, prob)
        if found_pos and found_neg:
            break

print(f"  positive case: {found_pos[0].name} p={found_pos[1]:.3f}")
print(f"  negative case: {found_neg[0].name} p={found_neg[1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, (path, prob), label, color in zip(
        axes, [found_neg, found_pos], ["DME Absent (true negative)", "DME Present (true positive)"],
        ["#10b981", "#ef4444"]):
    img = Image.open(path).convert("RGB").resize((260, 260))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"Real test image\n{label}\nmodel probability: {prob*100:.1f}%", fontsize=9, color=color)
fig.suptitle("Real held-out DME test images, both correctly classified by the trained model", fontsize=10)
fig.tight_layout()
fig.savefig("paper/fig_dme_pos_neg_real.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_dme_pos_neg_real.png")

# ============================================================
# SECTION F: real example lesion patches used to train each
# TCAV concept probe
# ============================================================
print("\nFinding real example patches per concept for display...")
seg_root = Path("data/idrid/segmentation")
concepts = ["microaneurysm", "hemorrhage", "exudate", "edema"]
examples = {}
rng = np.random.RandomState(7)

for split in ["DR_Training_Set"]:
    img_dir = seg_root / split / "Fundus Images"
    mask_dir = seg_root / split / "Combined Masks"
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue
        mask_arr = np.array(Image.open(mask_path).convert("RGB"))
        w, h = Image.open(img_path).size
        cx, cy = w / 2, h / 2
        center_r = 0.35 * min(w, h)

        if "microaneurysm" not in examples:
            c = _blob_centroids(mask_arr, IDRID_COLORS["microaneurysm"])
            if c: examples["microaneurysm"] = (img_path, c[len(c)//2])
        if "hemorrhage" not in examples:
            c = _blob_centroids(mask_arr, IDRID_COLORS["hemorrhage"])
            if c: examples["hemorrhage"] = (img_path, c[len(c)//2])
        if "exudate" not in examples:
            c = _blob_centroids(mask_arr, IDRID_COLORS["exudate"])
            if c: examples["exudate"] = (img_path, c[0])
        if "edema" not in examples:
            c = _blob_centroids(mask_arr, IDRID_COLORS["exudate"])
            for cc in c:
                if np.hypot(cc[0]-cx, cc[1]-cy) <= center_r:
                    examples["edema"] = (img_path, cc)
                    break
        if len(examples) == 4:
            break
    if len(examples) == 4:
        break

print("  found real examples for:", list(examples.keys()))

patch_size = 640
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, concept in zip(axes, concepts):
    img_path, (px, py) = examples[concept]
    fundus = Image.open(img_path).convert("RGB")
    fw, fh = fundus.size
    left = int(np.clip(px - patch_size/2, 0, fw - patch_size))
    top = int(np.clip(py - patch_size/2, 0, fh - patch_size))
    crop = fundus.crop((left, top, left + patch_size, top + patch_size))
    ax.imshow(crop); ax.axis("off")
    # mark the real blob centroid this crop is centered on - an annotation
    # of where the lesion is, not an alteration of any real pixel data
    cx_local, cy_local = px - left, py - top
    circ = plt.Circle((cx_local, cy_local), 45, fill=False, edgecolor="#22c55e", linewidth=2.2)
    ax.add_patch(circ)
    ax.set_title(concept, fontsize=10)
fig.suptitle("Real example lesion patches, cropped from real IDRiD segmentation masks, used to train each TCAV concept probe", fontsize=9.5)
fig.tight_layout()
fig.savefig("paper/fig_tcav_real_patches.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_tcav_real_patches.png")
