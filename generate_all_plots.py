import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from models.vqc_only import VQCOnly

IMG_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

class GenericDataset(Dataset):
    def __init__(self, df, img_index, transform):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_index[str(row["id_code"])]).convert("RGB")
        return self.transform(img), int(row["diagnosis"])

def index_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(p.stem, p)
    return idx

def load_ext_dataset(cfg, root, csv_name):
    root = Path(root)
    df = pd.read_csv(root / csv_name)
    df["id_code"] = df["id_code"].astype(str)
    df["id_code"] = df["id_code"].apply(lambda x: Path(x).stem)
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    idx = index_images(root)
    df = df[df["id_code"].isin(idx.keys())].reset_index(drop=True)
    transform = build_transforms(cfg, train=False)
    ds = GenericDataset(df, idx, transform)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0)
    return loader

def get_preds(model, loader, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), 1).argmax(1).cpu().numpy()
            preds.extend(p); trues.extend(y.numpy())
    return np.array(trues), np.array(preds)

def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["G0","G1","G2","G3","G4"],
                yticklabels=["G0","G1","G2","G3","G4"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    out = "experiments/results"

    # Load all models
    vit = ViTClassifier(nc).to(device)
    vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    vit.eval()

    hybrid = QuantumRetinaX(cfg).to(device)
    hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    hybrid.eval()

    vqc = VQCOnly(cfg).to(device)
    vqc.load_state_dict(torch.load("weights/vqc_only.pth", weights_only=True))
    vqc.eval()

    _, _, te_aptos, _ = build_aptos_loaders(cfg, num_workers=0)

    # === 1. VQC-only ablation confusion matrix ===
    print("\n[1/6] VQC-only confusion matrix...")
    y_true, y_pred = get_preds(vqc, te_aptos, device)
    plot_cm(y_true, y_pred, "VQC Only (Config B) - Confusion Matrix",
            f"{out}/vqc_only_confusion_matrix.png")

    # === 2. IDRiD confusion matrices (both models) ===
    print("\n[2/6] IDRiD confusion matrices...")
    idrid_loader = load_ext_dataset(cfg, "data/idrid", "idrid_labels.csv")

    y_true, y_pred = get_preds(vit, idrid_loader, device)
    plot_cm(y_true, y_pred, "ViT Baseline on IDRiD (unseen)",
            f"{out}/idrid_vit_confusion_matrix.png")

    y_true, y_pred = get_preds(hybrid, idrid_loader, device)
    plot_cm(y_true, y_pred, "ViT+VQC Hybrid on IDRiD (unseen)",
            f"{out}/idrid_quantum_confusion_matrix.png")

    # === 3. Messidor confusion matrices (both models) ===
    print("\n[3/6] Messidor confusion matrices...")
    mess_loader = load_ext_dataset(cfg, "data/messidor", "messidor_data.csv")

    y_true, y_pred = get_preds(vit, mess_loader, device)
    plot_cm(y_true, y_pred, "ViT Baseline on Messidor (unseen)",
            f"{out}/messidor_vit_confusion_matrix.png")

    y_true, y_pred = get_preds(hybrid, mess_loader, device)
    plot_cm(y_true, y_pred, "ViT+VQC Hybrid on Messidor (unseen)",
            f"{out}/messidor_quantum_confusion_matrix.png")

    # === 4. Cross-dataset accuracy comparison ===
    print("\n[4/6] Cross-dataset accuracy comparison chart...")
    datasets = ["APTOS\n(trained)", "Messidor\n(France)", "IDRiD\n(India)", "EyePACS\n(USA)"]
    vit_acc = [0.784, 0.594, 0.429, 0.642]
    vqc_acc = [0.816, 0.583, 0.435, 0.630]
    vit_kap = [0.881, 0.449, 0.578, 0.412]
    vqc_kap = [0.872, 0.430, 0.618, 0.412]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(4); w = 0.35
    ax1.bar(x - w/2, vit_acc, w, label="ViT-only", color="#4A90D9")
    ax1.bar(x + w/2, vqc_acc, w, label="ViT+VQC", color="#7F4FBF")
    ax1.set_xticks(x); ax1.set_xticklabels(datasets, fontsize=9)
    ax1.set_ylabel("Accuracy"); ax1.set_title("Cross-Dataset Accuracy")
    ax1.legend(); ax1.set_ylim(0, 1.0); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - w/2, vit_kap, w, label="ViT-only", color="#4A90D9")
    ax2.bar(x + w/2, vqc_kap, w, label="ViT+VQC", color="#7F4FBF")
    ax2.set_xticks(x); ax2.set_xticklabels(datasets, fontsize=9)
    ax2.set_ylabel("Kappa"); ax2.set_title("Cross-Dataset Kappa")
    ax2.legend(); ax2.set_ylim(0, 1.0); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Generalisation (trained on APTOS only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/crossdataset_full_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/crossdataset_full_comparison.png")

    # === 5. Training curves (from logged data) ===
    print("\n[5/6] Training curves...")
    vit_epochs = list(range(1, 21))
    vit_loss = [1.179, 0.890, 0.766, 0.677, 0.597, 0.498, 0.411, 0.330,
                0.253, 0.214, 0.174, 0.213, 0.162, 0.135, 0.115, 0.099,
                0.085, 0.063, 0.050, 0.084]
    vit_kappa_ep = [0.799, 0.864, 0.828, 0.817, 0.860, 0.876, 0.871, 0.880,
                    0.870, 0.858, 0.862, 0.869, 0.896, 0.812, 0.851, 0.859,
                    0.872, 0.876, 0.872, 0.865]
    vit_g1_ep = [0.436, 0.455, 0.764, 0.727, 0.545, 0.636, 0.636, 0.727,
                 0.727, 0.545, 0.709, 0.582, 0.800, 0.782, 0.745, 0.727,
                 0.600, 0.636, 0.564, 0.491]

    hyb_epochs = list(range(1, 9))
    hyb_loss = [1.057, 0.713, 0.563, 0.504, 0.435, 0.390, 0.354, 0.305]
    hyb_kappa_ep = [0.784, 0.823, 0.856, 0.864, 0.871, 0.871, 0.873, 0.884]
    hyb_g1_ep = [0.745, 0.727, 0.691, 0.691, 0.673, 0.709, 0.673, 0.673]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(vit_epochs, vit_loss, "b-o", markersize=3, label="ViT-only")
    axes[0].plot(hyb_epochs, hyb_loss, "purple", marker="s", markersize=3, label="ViT+VQC")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(vit_epochs, vit_kappa_ep, "b-o", markersize=3, label="ViT-only")
    axes[1].plot(hyb_epochs, hyb_kappa_ep, "purple", marker="s", markersize=3, label="ViT+VQC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Kappa")
    axes[1].set_title("Validation Kappa"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(vit_epochs, vit_g1_ep, "b-o", markersize=3, label="ViT-only")
    axes[2].plot(hyb_epochs, hyb_g1_ep, "purple", marker="s", markersize=3, label="ViT+VQC")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("G1-Sensitivity")
    axes[2].set_title("Grade-1 Sensitivity (Early Detection)"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("Training Curves: ViT-only vs ViT+VQC", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/training_curves.png")

    # === 6. Per-grade sensitivity comparison ===
    print("\n[6/6] Per-grade sensitivity comparison...")
    grades = ["G0\nHealthy", "G1\nMild", "G2\nModerate", "G3\nSevere", "G4\nProliferative"]
    vit_sens = [0.974, 0.732, 0.587, 0.724, 0.386]
    hyb_sens = [0.974, 0.554, 0.807, 0.414, 0.477]
    g1b_sens = [0.970, 0.786, 0.513, 0.000, 0.591]

    x = np.arange(5); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, vit_sens, w, label="ViT-only", color="#4A90D9")
    ax.bar(x, hyb_sens, w, label="ViT+VQC", color="#7F4FBF")
    ax.bar(x + w, g1b_sens, w, label="ViT+VQC (G1-boost)", color="#E74C3C", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_ylabel("Sensitivity (Recall)")
    ax.set_title("Per-Grade Sensitivity: Where Each Model Excels")
    ax.legend(); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/per_grade_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/per_grade_sensitivity.png")

    print("\n" + "=" * 50)
    print("ALL VISUALIZATIONS GENERATED:")
    print("  - vqc_only_confusion_matrix.png")
    print("  - idrid_vit_confusion_matrix.png")
    print("  - idrid_quantum_confusion_matrix.png")
    print("  - messidor_vit_confusion_matrix.png")
    print("  - messidor_quantum_confusion_matrix.png")
    print("  - crossdataset_full_comparison.png")
    print("  - training_curves.png")
    print("  - per_grade_sensitivity.png")
    print("  + previous: ablation_chart, model_comparison,")
    print("    crossdataset_g1_comparison, vit/quantum confusion matrices")
    print("=" * 50)

if __name__ == "__main__":
    main()
