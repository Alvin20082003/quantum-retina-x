import torch
import numpy as np
import json
import pickle
import cv2
from PIL import Image
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utils.config import load_config
from utils.seed import set_seed
from data.transforms import build_transforms
from models.hybrid import QuantumRetinaX

def extract_features(model, images, device):
    """Run images through frozen ViT backbone, return CLS token features."""
    features = []
    model.eval()
    with torch.no_grad():
        for img_tensor in images:
            x = img_tensor.unsqueeze(0).to(device)
            out = model.backbone(pixel_values=x)
            cls = out.last_hidden_state[:, 0].cpu().numpy()
            features.append(cls[0])
    return np.array(features)

# Color coding of the IDRiD "Combined Masks" (empirically determined, since
# this repackaged combined-mask format has no published legend): magenta is
# present in every single image at a consistent size -> optic disc (always
# present, not a lesion). Yellow is rare -> soft exudates (a genuinely rare
# lesion type). Of the remaining two, blue has far more numerous, smaller
# blobs -> microaneurysms; green has fewer but much larger, irregular blobs
# -> haemorrhages. That leaves red -> hard exudates.
IDRID_COLORS = {
    "microaneurysm": (0, 0, 255),
    "hemorrhage": (0, 255, 0),
    "exudate": (255, 0, 0),
    "soft_exudate": (255, 255, 0),
    "optic_disc": (255, 0, 255),
}

def _blob_centroids(mask_arr, color, min_area=10):
    m = np.all(mask_arr == np.array(color), axis=-1).astype(np.uint8)
    if m.sum() == 0:
        return []
    n, _, cstats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [tuple(centroids[i]) for i in range(1, n) if cstats[i, cv2.CC_STAT_AREA] >= min_area]

def extract_idrid_concept_patches(seg_root, transform, patch_size=640, n_per_concept=60, seed=42):
    """Real concept patches cropped from IDRiD's pixel-level lesion masks,
    instead of whole APTOS images bucketed by DR grade.

    - "microaneurysm" and "hemorrhage" are separate concepts (each from its
      own mask color), not pooled together, so each gets its own
      independently-verifiable detector.
    - "exudate" concept = hard + soft exudate blobs ("bright lesions")
    - "edema" concept: IDRiD has no direct macular-edema segmentation, but
      DME risk is clinically graded by hard-exudate proximity to the
      macula, so this uses hard-exudate blobs near the image center as a
      real, clinically-grounded proxy instead of a whole-image grade guess.
    - "healthy" = random crops containing no lesion/disc pixels and with
      real (non-black-background) tissue brightness.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for split in ["DR_Training_Set", "DR_Testing_Set"]:
        img_dir = Path(seg_root) / split / "Fundus Images"
        mask_dir = Path(seg_root) / split / "Combined Masks"
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                pairs.append((img_path, mask_path))

    buckets = {"microaneurysm": [], "hemorrhage": [], "exudate": [], "edema": [], "healthy": []}

    for img_path, mask_path in pairs:
        fundus_probe = Image.open(img_path).convert("RGB")
        mask_arr = np.array(Image.open(mask_path).convert("RGB"))
        gray = np.array(fundus_probe.convert("L"))
        w, h = fundus_probe.size
        cx, cy = w / 2, h / 2
        center_r = 0.35 * min(w, h)
        any_mark = mask_arr.any(axis=-1)

        ma = _blob_centroids(mask_arr, IDRID_COLORS["microaneurysm"])
        for c in ma:
            buckets["microaneurysm"].append((img_path, c))
        he = _blob_centroids(mask_arr, IDRID_COLORS["hemorrhage"])
        for c in he:
            buckets["hemorrhage"].append((img_path, c))

        ex = _blob_centroids(mask_arr, IDRID_COLORS["exudate"])
        se = _blob_centroids(mask_arr, IDRID_COLORS["soft_exudate"])
        for c in ex + se:
            buckets["exudate"].append((img_path, c))
        for c in ex:
            if np.hypot(c[0] - cx, c[1] - cy) <= center_r:
                buckets["edema"].append((img_path, c))

        found = tries = 0
        while found < 5 and tries < 30:
            tries += 1
            ry = rng.randint(patch_size, h - patch_size)
            rx = rng.randint(patch_size, w - patch_size)
            region = any_mark[ry - patch_size//2:ry + patch_size//2, rx - patch_size//2:rx + patch_size//2]
            bright = gray[ry - patch_size//2:ry + patch_size//2, rx - patch_size//2:rx + patch_size//2]
            if not region.any() and bright.mean() > 40:
                buckets["healthy"].append((img_path, (rx, ry)))
                found += 1

    print("  concept pool sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    def to_patches(items, n):
        idxs = rng.permutation(len(items))[:n]
        patches = []
        cache = {}
        for i in idxs:
            img_path, (px, py) = items[i]
            if img_path not in cache:
                cache[img_path] = Image.open(img_path).convert("RGB")
            fundus = cache[img_path]
            fw, fh = fundus.size
            left = int(np.clip(px - patch_size/2, 0, fw - patch_size))
            top = int(np.clip(py - patch_size/2, 0, fh - patch_size))
            crop = fundus.crop((left, top, left + patch_size, top + patch_size))
            patches.append(transform(crop))
        return patches

    return {k: to_patches(v, n_per_concept) for k, v in buckets.items()}

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QuantumRetinaX(cfg).to(device)
    model.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    model.eval()

    transform = build_transforms(cfg, train=False)
    out = "experiments/results"
    Path(f"{out}/tcav").mkdir(parents=True, exist_ok=True)

    # Real concept patches cropped from IDRiD's pixel-level lesion masks
    # (data/idrid/segmentation/) instead of whole APTOS images bucketed by
    # DR grade. See extract_idrid_concept_patches() for the lesion-type ->
    # concept mapping and reasoning.
    print("Extracting real lesion patches from IDRiD segmentation masks...")
    concept_patches = extract_idrid_concept_patches("data/idrid/segmentation", transform)

    neg_images = concept_patches["healthy"]
    print(f"Negative (healthy) patches: {len(neg_images)}")
    neg_features = extract_features(model, neg_images, device)
    print(f"Negative features shape: {neg_features.shape}")

    # Still use APTOS (grade-labelled, no pixel masks) to see how each
    # concept's presence trends across DR severity for the summary charts.
    import pandas as pd
    root = Path("data/archive")
    df = pd.read_csv(root / "train.csv")
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            idx[p.stem] = p

    tcav_results = {}

    for concept_name in ["microaneurysm", "hemorrhage", "exudate", "edema"]:
        print(f"\n--- Concept: {concept_name} ---")
        pos_images = concept_patches[concept_name]
        print(f"  Positive patches: {len(pos_images)}")
        pos_features = extract_features(model, pos_images, device)

        # Train concept probe (linear classifier)
        X = np.vstack([pos_features, neg_features])
        y = np.array([1]*len(pos_features) + [0]*len(neg_features))
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X, y)
        probe_acc = probe.score(X, y)
        print(f"  Probe accuracy: {probe_acc:.3f}")

        # Save the trained probe so the backend can score new images live
        # (predict_proba on that image's own features) instead of relying
        # on this offline, grade-bucketed table.
        probe_path = Path(f"{out}/tcav/probe_{concept_name}.pkl")
        with open(probe_path, "wb") as pf:
            pickle.dump(probe, pf)
        print(f"  saved {probe_path}")

        # CAV = the weight vector of the probe
        cav = probe.coef_[0]
        cav = cav / np.linalg.norm(cav)

        # Compute TCAV score: for test images of each grade,
        # what fraction have positive directional derivative toward this concept?
        tcav_by_grade = {}
        for grade in range(5):
            grade_rows = df[df["diagnosis"] == grade].sample(
                n=min(30, len(df[df["diagnosis"] == grade])), random_state=42)
            grade_images = []
            for _, row in grade_rows.iterrows():
                if row["id_code"] in idx:
                    img = Image.open(idx[row["id_code"]]).convert("RGB")
                    grade_images.append(transform(img))

            grade_features = extract_features(model, grade_images, device)

            # Directional derivative: dot product of features with CAV
            dots = grade_features @ cav
            tcav_score = float(np.mean(dots > 0))
            tcav_by_grade[grade] = tcav_score

        tcav_results[concept_name] = tcav_by_grade
        print(f"  TCAV scores by grade: {tcav_by_grade}")

    # Save TCAV scores
    with open(f"{out}/tcav/tcav_scores.json", "w") as f:
        json.dump(tcav_results, f, indent=2)
    print(f"\n[ok] saved tcav_scores.json")

    # Save as readable text
    with open(f"{out}/tcav/tcav_scores.txt", "w") as f:
        f.write("TCAV Concept Scores by DR Grade\n")
        f.write("=" * 50 + "\n\n")
        for concept, scores in tcav_results.items():
            f.write(f"{concept.upper()}:\n")
            for grade, score in scores.items():
                label = ["Healthy", "Mild (Early)", "Moderate", "Severe", "Proliferative"][grade]
                f.write(f"  Grade {grade} ({label}): {score:.3f}\n")
            f.write("\n")
    print(f"[ok] saved tcav_scores.txt")

    # Plot TCAV bar chart
    grades = ["G0\nHealthy", "G1\nMild", "G2\nModerate", "G3\nSevere", "G4\nProliferative"]
    x = np.arange(5)
    n_concepts = len(tcav_results)
    w = 0.8 / n_concepts
    colors = {"microaneurysm": "#8E44AD", "hemorrhage": "#E24B4A", "exudate": "#EF9F27", "edema": "#378ADD"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (concept, scores) in enumerate(tcav_results.items()):
        vals = [scores[g] for g in range(5)]
        ax.bar(x + i*w, vals, w, label=concept.capitalize(), color=colors[concept])

    ax.set_xticks(x + w*(n_concepts-1)/2); ax.set_xticklabels(grades)
    ax.set_ylabel("TCAV Score")
    ax.set_title("Concept Influence (TCAV) Across DR Grades")
    ax.legend(); ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/tcav/tcav_scores_chart.png", dpi=150)
    plt.close(fig)
    print(f"[ok] saved tcav_scores_chart.png")

    # Plot concept influence summary for dashboard
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    concept_names = list(tcav_results.keys())
    # Average TCAV score across diseased grades (1-4)
    avg_scores = []
    for concept in concept_names:
        scores = tcav_results[concept]
        avg = np.mean([scores[g] for g in [1, 2, 3, 4]])
        avg_scores.append(avg)

    bars = ax2.barh(concept_names, avg_scores,
                     color=[colors[c] for c in concept_names], height=0.5)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Average TCAV Score (diseased grades)")
    ax2.set_title("Why the AI made its decision ? concept influence")
    for bar, val in zip(bars, avg_scores):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{out}/tcav/tcav_dashboard_summary.png", dpi=150)
    plt.close(fig2)
    print(f"[ok] saved tcav_dashboard_summary.png")

    print("\n" + "=" * 50)
    print("PHASE 7 ? TCAV COMPLETE")
    print("All outputs in experiments/results/tcav/")
    print("=" * 50)

if __name__ == "__main__":
    main()
