"""
Faithfulness validation: does Grad-CAM actually point at real lesions?

Compares the model's Grad-CAM heatmap against expert-drawn pixel-level lesion
masks from the official IDRiD Segmentation subset (81 images: 54 train + 27
test, data/idrid/segmentation/). This is the only IDRiD data with real,
per-pixel ground truth for where disease actually is, so it's the only way to
put a real number on "is the explanation trustworthy" instead of eyeballing it.

Mask color coding (Combined Masks, verified empirically): magenta (255,0,255)
is the optic disc (present in every image, not a lesion) and is excluded.
Every other non-black color (red/green/blue/yellow = microaneurysms,
haemorrhages, hard/soft exudates) counts as ground-truth lesion area.
"""
import json
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.transforms import build_transforms
from models.backbone import ViTClassifier

OPTIC_DISC_COLOR = (255, 0, 255)


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, result.size(2))
    return result.permute(0, 3, 1, 2)


def make_fundus_mask(img_uint8, erode_px=9):
    """Same fundus masking used in backend/main.py, kept in sync manually
    since it's a small, stable helper (see backend/main.py for the live copy)."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    mask = (gray > 15).astype(np.uint8)
    kernel = np.ones((erode_px, erode_px), np.uint8)
    return cv2.erode(mask, kernel, iterations=1).astype(bool)


def load_gt_lesion_mask(mask_path, size=224):
    mask_img = Image.open(mask_path).convert("RGB").resize((size, size), Image.NEAREST)
    arr = np.array(mask_img)
    non_black = arr.any(axis=-1)
    optic_disc = np.all(arr == np.array(OPTIC_DISC_COLOR), axis=-1)
    return non_black & ~optic_disc


def cam_to_binary(gray_cam, fg_mask, lo_pct=60, hi_pct=99.5, thresh=0.5):
    fg_vals = gray_cam[fg_mask]
    if fg_vals.size == 0:
        return np.zeros_like(fg_mask)
    lo = np.percentile(fg_vals, lo_pct)
    hi = np.percentile(fg_vals, hi_pct)
    stretched = np.clip((gray_cam - lo) / max(hi - lo, 1e-8), 0, 1)
    stretched = np.where(fg_mask, stretched, 0)
    return stretched > thresh


def overlap_metrics(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    gt_area = gt.sum()
    pred_area = pred.sum()
    iou = inter / union if union > 0 else float("nan")
    recall = inter / gt_area if gt_area > 0 else float("nan")   # of real lesion, how much did heatmap cover
    precision = inter / pred_area if pred_area > 0 else float("nan")  # of heatmap, how much was real lesion
    return {"iou": float(iou), "recall": float(recall), "precision": float(precision),
            "gt_area_frac": float(gt_area / gt.size), "pred_area_frac": float(pred_area / pred.size)}


def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Matches backend/main.py: Grad-CAM is computed on the classical ViT head,
    # not QuantumRetinaX, because QuantumRetinaX.forward() wraps the backbone
    # in torch.no_grad() (it's frozen), which blocks gradients Grad-CAM needs
    # at this layer. This validates exactly what the live dashboard shows.
    nc = cfg["data"]["num_classes"]
    model = ViTClassifier(nc).to(device)
    model.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    model.eval()

    target_layer = model.backbone.encoder.layer[-1].layernorm_before
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    transform = build_transforms(cfg, train=False)

    seg_root = Path("data/idrid/segmentation")
    pairs = []
    for split in ["DR_Training_Set", "DR_Testing_Set"]:
        img_dir = seg_root / split / "Fundus Images"
        mask_dir = seg_root / split / "Combined Masks"
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                pairs.append((img_path, mask_path))
    print(f"[faithfulness] found {len(pairs)} image/mask pairs")

    results = []
    for img_path, mask_path in pairs:
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        img_uint8 = np.array(img_resized)

        tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), 1).cpu().numpy()[0]
        grade = int(np.argmax(probs))

        gray_cam = cam(input_tensor=tensor)[0]
        fg_mask = make_fundus_mask(img_uint8)
        pred_mask = cam_to_binary(gray_cam, fg_mask)
        gt_mask = load_gt_lesion_mask(mask_path)

        m = overlap_metrics(gt_mask, pred_mask)
        m.update({"id": img_path.stem, "grade": grade})
        results.append(m)
        print(f"  {img_path.stem} grade={grade} iou={m['iou']:.3f} "
              f"recall={m['recall']:.3f} precision={m['precision']:.3f}")

    out = Path("experiments/results/faithfulness")
    out.mkdir(parents=True, exist_ok=True)

    def safe_mean(key, rows):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    overall = {"n": len(results),
               "mean_iou": safe_mean("iou", results),
               "mean_recall": safe_mean("recall", results),
               "mean_precision": safe_mean("precision", results)}

    by_grade = {}
    for g in range(5):
        rows = [r for r in results if r["grade"] == g]
        if rows:
            by_grade[g] = {"n": len(rows), "mean_iou": safe_mean("iou", rows),
                           "mean_recall": safe_mean("recall", rows),
                           "mean_precision": safe_mean("precision", rows)}

    with open(out / "faithfulness_results.json", "w") as f:
        json.dump({"overall": overall, "by_grade": by_grade, "per_image": results}, f, indent=2)

    with open(out / "faithfulness_summary.txt", "w") as f:
        f.write("Grad-CAM Faithfulness Validation vs IDRiD Expert Lesion Masks\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Images evaluated: {overall['n']}\n")
        f.write(f"Mean IoU:       {overall['mean_iou']:.3f}\n")
        f.write(f"Mean Recall:    {overall['mean_recall']:.3f}  (fraction of real lesion area the heatmap covers)\n")
        f.write(f"Mean Precision: {overall['mean_precision']:.3f}  (fraction of highlighted area that is real lesion)\n\n")
        f.write("By predicted grade:\n")
        for g, s in sorted(by_grade.items()):
            f.write(f"  Grade {g} (n={s['n']}): IoU={s['mean_iou']:.3f} "
                    f"Recall={s['mean_recall']:.3f} Precision={s['mean_precision']:.3f}\n")
    print(f"[ok] saved {out}/faithfulness_summary.txt")

    grades = sorted(by_grade.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(grades)); w = 0.25
    ax.bar(x - w, [by_grade[g]["mean_iou"] for g in grades], w, label="IoU", color="#8b5cf6")
    ax.bar(x,     [by_grade[g]["mean_recall"] for g in grades], w, label="Recall", color="#3b82f6")
    ax.bar(x + w, [by_grade[g]["mean_precision"] for g in grades], w, label="Precision", color="#10b981")
    ax.set_xticks(x); ax.set_xticklabels([f"Grade {g}\n(n={by_grade[g]['n']})" for g in grades])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("Grad-CAM Faithfulness vs Expert Lesion Masks (IDRiD Segmentation subset)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "faithfulness_chart.png", dpi=150)
    plt.close(fig)
    print(f"[ok] saved {out}/faithfulness_chart.png")

    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="faithfulness_idrid", dataset="idrid_segmentation", model="ViT-only (Grad-CAM)",
        notes=f"n={overall['n']} mean_iou={overall['mean_iou']:.3f} "
              f"mean_recall={overall['mean_recall']:.3f} mean_precision={overall['mean_precision']:.3f}")
    print("\n" + "=" * 60)
    print(f"OVERALL: IoU={overall['mean_iou']:.3f} Recall={overall['mean_recall']:.3f} "
          f"Precision={overall['mean_precision']:.3f}")
    print("Faithfulness validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
