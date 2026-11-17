"""
EyePACS-only cross-dataset evaluation (fast version).
Uses batch_size=32 and pre-indexes images efficiently.
Saves predictions CSV and appends to cross_dataset_summary.csv.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.config import load_config
from utils.seed import set_seed
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OUT_DIR = Path("experiments/results/cross_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH = 32


class EyePACSDataset(Dataset):
    def __init__(self, df, img_index, transform):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.img_index[str(row["id_code"])]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(row["diagnosis"]), str(row["id_code"])


def build_index(root):
    print("  Building image index (may take a moment)...")
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx[p.stem] = p
    print(f"  Index built: {len(idx)} images")
    return idx


@torch.no_grad()
def evaluate(model, loader, device, nc=5):
    model.eval()
    all_probs, all_trues, all_ids = [], [], []
    n = len(loader)
    for i, batch in enumerate(loader):
        x, y, ids = batch
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        all_probs.append(p)
        all_trues.extend(y.numpy())
        all_ids.extend(ids)
        if i % 200 == 0:
            print(f"    batch {i}/{n}...", flush=True)
    probs = np.concatenate(all_probs)
    trues = np.array(all_trues)
    metrics = compute_metrics(trues, probs, nc)
    return metrics, probs, trues, all_ids


def save_predictions(model_name, ds_name, ids, trues, probs, nc):
    preds = probs.argmax(1)
    rows = []
    for i, (img_id, true_label, pred_label) in enumerate(zip(ids, trues, preds)):
        row = {
            "image_id": img_id,
            "true_label": int(true_label),
            "predicted_label": int(pred_label),
        }
        for c in range(nc):
            row[f"prob_grade{c}"] = round(float(probs[i, c]), 6)
        rows.append(row)
    df = pd.DataFrame(rows)
    safe_name = model_name.replace("+", "_").replace(" ", "_")
    out_path = OUT_DIR / f"predictions_{safe_name}_{ds_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(df)} rows)")
    return out_path


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    print(f"Device: {device}")

    # Load models
    vit = ViTClassifier(nc).to(device)
    vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    vit.eval()
    print("Loaded: weights/vit_baseline.pth")

    hybrid = QuantumRetinaX(cfg).to(device)
    hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    hybrid.eval()
    print("Loaded: weights/hybrid_vit_vqc.pth")

    models = {"ViT-only": vit, "ViT+VQC": hybrid}

    # Load CSV
    root = Path("data/eyepacs")
    df = pd.read_csv(root / "trainLabels.csv", usecols=[0, 1])
    df.columns = ["id_code", "diagnosis"]
    df["id_code"] = df["id_code"].astype(str).apply(lambda x: Path(x).stem)
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)

    idx = build_index(root)
    before = len(df)
    df = df[df["id_code"].isin(idx.keys())].reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} unmatched rows")

    n = len(df)
    grade_dist = dict(sorted(df["diagnosis"].value_counts().items()))
    print(f"  EyePACS: {n} images | grades: {grade_dist}")

    transform = build_transforms(cfg, train=False)
    ds = EyePACSDataset(df, idx, transform)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    summary_rows = []

    print("\n=== EyePACS Evaluation ===")
    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")
        metrics, probs, trues, ids = evaluate(model, loader, device, nc)

        acc   = round(metrics["accuracy"],  4)
        f1    = round(metrics["macro_f1"],  4)
        auc   = round(metrics["auc_roc"],   4)
        kappa = round(metrics["kappa"],     4)
        g1    = round(metrics["grade1_sensitivity"], 4)

        print(f"  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"Kappa={kappa:.4f}  G1-sens={g1:.4f}")

        save_predictions(model_name, "EyePACS", ids, trues, probs, nc)

        summary_rows.append({
            "Training dataset": "APTOS (India)",
            "Evaluation dataset": "EyePACS (USA)",
            "Model": model_name,
            "Test n": n,
            "Accuracy": acc,
            "Macro-F1": f1,
            "AUC": auc,
            "Kappa": kappa,
            "G1-sensitivity": g1,
        })

    # Append to existing summary or create new
    summary_path = OUT_DIR / "cross_dataset_summary.csv"
    new_df = pd.DataFrame(summary_rows)

    # Load existing IDRiD + Messidor rows if present
    existing_rows = [
        {"Training dataset": "APTOS (India)", "Evaluation dataset": "IDRiD (India)",
         "Model": "ViT-only", "Test n": 455,
         "Accuracy": 0.4286, "Macro-F1": 0.3252, "AUC": 0.7861, "Kappa": 0.5781, "G1-sensitivity": 0.3182},
        {"Training dataset": "APTOS (India)", "Evaluation dataset": "IDRiD (India)",
         "Model": "ViT+VQC", "Test n": 455,
         "Accuracy": 0.4352, "Macro-F1": 0.3354, "AUC": 0.6572, "Kappa": 0.6184, "G1-sensitivity": 0.5000},
        {"Training dataset": "APTOS (India)", "Evaluation dataset": "Messidor (France)",
         "Model": "ViT-only", "Test n": 1744,
         "Accuracy": 0.5940, "Macro-F1": 0.3234, "AUC": 0.7598, "Kappa": 0.4492, "G1-sensitivity": 0.0333},
        {"Training dataset": "APTOS (India)", "Evaluation dataset": "Messidor (France)",
         "Model": "ViT+VQC", "Test n": 1744,
         "Accuracy": 0.5831, "Macro-F1": 0.2953, "AUC": 0.5770, "Kappa": 0.4302, "G1-sensitivity": 0.0556},
    ]
    all_rows = existing_rows + summary_rows
    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FULL CROSS-DATASET SUMMARY")
    print("=" * 70)
    print(full_df.to_string(index=False))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
