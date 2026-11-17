"""
Cross-dataset evaluation for QuantumRetina-X
Trains on APTOS, tests on IDRiD, Messidor, EyePACS (zero-shot generalisation).
Saves:
  - experiments/results/cross_dataset/predictions_{model}_{dataset}.csv
  - experiments/results/cross_dataset/cross_dataset_summary.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
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

IMG_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
OUT_DIR = Path("experiments/results/cross_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class GenericDataset(Dataset):
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


def index_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx[p.stem] = p
    return idx


def load_dataset(cfg, root, csv_name):
    root = Path(root)
    df = pd.read_csv(root / csv_name, usecols=[0, 1])
    df.columns = ["id_code", "diagnosis"]
    df["id_code"] = df["id_code"].astype(str).apply(lambda x: Path(x).stem)
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    idx = index_images(root)
    before = len(df)
    df = df[df["id_code"].isin(idx.keys())].reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with no matching image")
    transform = build_transforms(cfg, train=False)
    ds = GenericDataset(df, idx, transform)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0, drop_last=False)
    grade_dist = dict(sorted(df["diagnosis"].value_counts().items()))
    print(f"  {len(df)} images | grades: {grade_dist}")
    return loader, len(df)


@torch.no_grad()
def evaluate(model, loader, device, nc=5):
    model.eval()
    all_probs, all_trues, all_ids = [], [], []
    for batch in loader:
        x, y, ids = batch
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        all_probs.append(p)
        all_trues.extend(y.numpy())
        all_ids.extend(ids)
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
    out_path = OUT_DIR / f"predictions_{model_name.replace('+', '_')}_{ds_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved {out_path}")
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
    print("Loaded weights/vit_baseline.pth")

    hybrid = QuantumRetinaX(cfg).to(device)
    hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    hybrid.eval()
    print("Loaded weights/hybrid_vit_vqc.pth")

    models = {"ViT-only": vit, "ViT+VQC": hybrid}

    datasets = {
        "IDRiD":    ("data/idrid",    "idrid_labels.csv"),
        "Messidor": ("data/messidor", "messidor_data.csv"),
        "EyePACS":  ("data/eyepacs",  "trainLabels.csv"),
    }

    summary_rows = []

    print("\n" + "=" * 70)
    print("CROSS-DATASET EVALUATION: APTOS-trained -> unseen datasets")
    print("=" * 70)

    for ds_name, (root, csv) in datasets.items():
        print(f"\n{'-'*50}")
        print(f"Dataset: {ds_name}")
        loader, n = load_dataset(cfg, root, csv)

        for model_name, model in models.items():
            print(f"  Running {model_name}...")
            metrics, probs, trues, ids = evaluate(model, loader, device, nc)

            acc   = round(metrics["accuracy"],  4)
            f1    = round(metrics["macro_f1"],  4)
            auc   = round(metrics["auc_roc"],   4)
            kappa = round(metrics["kappa"],     4)
            g1    = round(metrics["grade1_sensitivity"], 4)

            print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
                  f"Kappa={kappa:.4f}  G1-sens={g1:.4f}")

            save_predictions(model_name, ds_name, ids, trues, probs, nc)

            summary_rows.append({
                "Training dataset": "APTOS (India)",
                "Evaluation dataset": ds_name,
                "Model": model_name,
                "Test n": n,
                "Accuracy": acc,
                "Macro-F1": f1,
                "AUC": auc,
                "Kappa": kappa,
                "G1-sensitivity": g1,
            })

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "cross_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
