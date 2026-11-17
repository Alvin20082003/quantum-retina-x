"""Real, model-generated comparison images for Sections D and E - no synthetic
or illustrative content. Everything here is a real image from the real APTOS
held-out test set, run through the real trained models."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.hybrid import QuantumRetinaX

cfg = load_config(); set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]

_, _, test_loader, info = build_aptos_loaders(cfg, num_workers=0)
test_df = test_loader.dataset.df
img_index = test_loader.dataset.img_index
print(f"Real held-out APTOS test set: {len(test_df)} images")

def load_model(path, ncls=5):
    m = QuantumRetinaX(cfg, num_classes=ncls).to(device)
    m.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    m.eval()
    return m

hybrid = load_model("weights/hybrid_vit_vqc.pth")
g1boost = load_model("weights/hybrid_g1boost.pth")

from data.transforms import build_transforms
transform = build_transforms(cfg, train=False)

# ============================================================
# PART 1 (Section E): find a REAL test image where the plain
# hybrid model got it wrong and the G1-boosted model got it right
# ============================================================
print("\nScanning real test set for a genuine plain-wrong / G1boost-right case...")
found = None
with torch.no_grad():
    for _, row in test_df.iterrows():
        true_grade = int(row["diagnosis"])
        if true_grade != 1:
            continue
        img_path = img_index[str(row["id_code"])]
        img_pil = Image.open(img_path).convert("RGB")
        tensor = transform(img_pil).unsqueeze(0).to(device)
        p_plain = torch.softmax(hybrid(tensor), 1)[0]
        p_boost = torch.softmax(g1boost(tensor), 1)[0]
        pred_plain = int(p_plain.argmax())
        pred_boost = int(p_boost.argmax())
        if pred_plain != 1 and pred_boost == 1:
            found = (img_path, true_grade, pred_plain, p_plain.cpu().numpy(),
                     pred_boost, p_boost.cpu().numpy())
            print(f"  found: {img_path.name}  true=G{true_grade}  "
                  f"plain_pred=G{pred_plain} ({p_plain[pred_plain]*100:.1f}%)  "
                  f"g1boost_pred=G{pred_boost} ({p_boost[pred_boost]*100:.1f}%)")
            break

if found:
    img_path, true_grade, pred_plain, probs_plain, pred_boost, probs_boost = found
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    axes[0].imshow(img); axes[0].axis("off")
    axes[0].set_title(f"Real test image\ntrue label: Grade {true_grade}", fontsize=10)

    labels = [f"G{i}" for i in range(5)]
    colors_plain = ["#ef4444" if i == pred_plain else "#334155" for i in range(5)]
    axes[1].bar(labels, probs_plain * 100, color=colors_plain)
    axes[1].set_title(f"Plain hybrid model\npredicted: Grade {pred_plain} (WRONG)", fontsize=10, color="#b91c1c")
    axes[1].set_ylim(0, 100)

    colors_boost = ["#10b981" if i == pred_boost else "#334155" for i in range(5)]
    axes[2].bar(labels, probs_boost * 100, color=colors_boost)
    axes[2].set_title(f"Grade-1-boosted model\npredicted: Grade {pred_boost} (CORRECT)", fontsize=10, color="#047857")
    axes[2].set_ylim(0, 100)

    fig.suptitle("Real case from the held-out test set where Grade-1 reweighting changed a wrong prediction to correct", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("paper/fig_g1boost_real_flip.png", dpi=150, bbox_inches="tight")
    print("saved paper/fig_g1boost_real_flip.png")
else:
    print("  No such case found in the real test set - will not fabricate one.")

# ============================================================
# PART 2 (Section D): 5 real test images, one per true grade,
# showing real reducer + VQC outputs side by side
# ============================================================
print("\nSelecting one real test image per grade for Section D comparison...")
examples = {}
for _, row in test_df.iterrows():
    g = int(row["diagnosis"])
    if g not in examples:
        examples[g] = row
    if len(examples) == 5:
        break

fig, axes = plt.subplots(3, 5, figsize=(13, 6.5),
                          gridspec_kw={"height_ratios": [1.3, 1, 1]})
with torch.no_grad():
    for col, g in enumerate(sorted(examples.keys())):
        row = examples[g]
        img_path = img_index[str(row["id_code"])]
        img_pil = Image.open(img_path).convert("RGB")
        tensor = transform(img_pil).unsqueeze(0).to(device)
        cls = hybrid.backbone(pixel_values=tensor).last_hidden_state[:, 0]
        reduced = hybrid.reducer(cls)[0].cpu().numpy()
        vqc_out = hybrid.vqc(hybrid.reducer(cls))[0].cpu().numpy()

        axes[0, col].imshow(img_pil.resize((160, 160))); axes[0, col].axis("off")
        axes[0, col].set_title(f"Real Grade {g}\nimage", fontsize=9)

        axes[1, col].bar(range(4), reduced, color=["#10b981" if v >= 0 else "#ef4444" for v in reduced])
        axes[1, col].axhline(0, color="#94a3b8", linewidth=0.6)
        axes[1, col].set_ylim(-3, 3)
        axes[1, col].set_xticks([]); axes[1, col].tick_params(labelsize=7)
        if col == 0: axes[1, col].set_ylabel("reducer\n(4 real values)", fontsize=8)

        axes[2, col].bar(range(4), vqc_out, color="#f59e0b")
        axes[2, col].axhline(0, color="#94a3b8", linewidth=0.6)
        axes[2, col].set_ylim(-1, 1)
        axes[2, col].set_xticks([]); axes[2, col].tick_params(labelsize=7)
        if col == 0: axes[2, col].set_ylabel("VQC output\n(4 real qubits)", fontsize=8)

        print(f"  Grade {g}: reducer={np.round(reduced,2)}  vqc={np.round(vqc_out,2)}")

fig.suptitle("Real reducer and VQC output compared across 5 real images, one per true grade (held-out APTOS test set)", fontsize=10)
fig.tight_layout()
fig.savefig("paper/fig_d_cross_grade_comparison.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_d_cross_grade_comparison.png")
