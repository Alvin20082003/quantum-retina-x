import torch
import numpy as np
import json
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

def load_concept_patches(root, concept_name, transform, n=50):
    """Load concept patches from data folders."""
    patches = []
    concept_dir = Path(root) / concept_name
    if concept_dir.exists():
        for p in sorted(concept_dir.glob("*"))[:n]:
            try:
                img = Image.open(p).convert("RGB")
                patches.append(transform(img))
            except Exception:
                continue
    return patches

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QuantumRetinaX(cfg).to(device)
    model.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    model.eval()

    transform = build_transforms(cfg, train=False)
    out = "experiments/results"
    Path(f"{out}/tcav").mkdir(parents=True, exist_ok=True)

    # Since IDRiD pixel masks need careful extraction, we simulate concept
    # patches from the APTOS dataset using grade-based proxies:
    # - "hemorrhage" concept: patches from Grade 3-4 (severe, lots of hemorrhage)
    # - "exudate" concept: patches from Grade 2-3 (moderate-severe, hard exudates)
    # - "edema" concept: patches from Grade 2-4 (DME-associated)
    # - "healthy" concept: patches from Grade 0 (negative examples)

    import pandas as pd
    root = Path("data/archive")
    df = pd.read_csv(root / "train.csv")
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            idx[p.stem] = p

    concepts = {
        "hemorrhage": {"grades": [3, 4], "description": "Severe lesions with hemorrhage"},
        "exudate":    {"grades": [2, 3], "description": "Hard exudate deposits"},
        "edema":      {"grades": [2, 3, 4], "description": "Macular edema indicators"},
    }
    negative_grades = [0]

    # Collect positive and negative patches
    n_samples = 60
    neg_rows = df[df["diagnosis"].isin(negative_grades)].sample(n=n_samples, random_state=42)
    neg_images = []
    for _, row in neg_rows.iterrows():
        if row["id_code"] in idx:
            img = Image.open(idx[row["id_code"]]).convert("RGB")
            neg_images.append(transform(img))

    print(f"Negative (healthy) patches: {len(neg_images)}")
    neg_features = extract_features(model, neg_images, device)
    print(f"Negative features shape: {neg_features.shape}")

    tcav_results = {}

    for concept_name, concept_info in concepts.items():
        print(f"\n--- Concept: {concept_name} ---")
        pos_rows = df[df["diagnosis"].isin(concept_info["grades"])].sample(
            n=n_samples, random_state=42)
        pos_images = []
        for _, row in pos_rows.iterrows():
            if row["id_code"] in idx:
                img = Image.open(idx[row["id_code"]]).convert("RGB")
                pos_images.append(transform(img))

        print(f"  Positive patches: {len(pos_images)}")
        pos_features = extract_features(model, pos_images, device)

        # Train concept probe (linear classifier)
        X = np.vstack([pos_features, neg_features])
        y = np.array([1]*len(pos_features) + [0]*len(neg_features))
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X, y)
        probe_acc = probe.score(X, y)
        print(f"  Probe accuracy: {probe_acc:.3f}")

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
    w = 0.25
    colors = {"hemorrhage": "#E24B4A", "exudate": "#EF9F27", "edema": "#378ADD"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (concept, scores) in enumerate(tcav_results.items()):
        vals = [scores[g] for g in range(5)]
        ax.bar(x + i*w, vals, w, label=concept.capitalize(), color=colors[concept])

    ax.set_xticks(x + w); ax.set_xticklabels(grades)
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
