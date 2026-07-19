import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX

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
    _, _, te, info = build_aptos_loaders(cfg, num_workers=0)
    out = "experiments/results"

    # ViT baseline confusion matrix
    vit = ViTClassifier(nc).to(device)
    vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    y_true, y_pred_vit = get_preds(vit, te, device)
    plot_cm(y_true, y_pred_vit, "ViT Baseline - Confusion Matrix",
            f"{out}/vit_confusion_matrix.png")

    # Hybrid confusion matrix
    hybrid = QuantumRetinaX(cfg).to(device)
    hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    y_true, y_pred_hyb = get_preds(hybrid, te, device)
    plot_cm(y_true, y_pred_hyb, "ViT+VQC Hybrid - Confusion Matrix",
            f"{out}/quantum_confusion_matrix.png")

    # Model comparison bar chart
    import pandas as pd
    results = {
        "Model": ["ViT-only", "ViT+VQC"],
        "Accuracy": [0.784, 0.816],
        "Kappa": [0.881, 0.872],
        "Macro-F1": [0.638, 0.659],
        "G1-Sensitivity": [0.732, 0.786],
        "AUC-ROC": [0.936, 0.894],
    }
    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Kappa", "Macro-F1", "G1-Sensitivity", "AUC-ROC"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x - w/2, [df[m][0] for m in metrics], w, label="ViT-only", color="#4A90D9")
    ax.bar(x + w/2, [df[m][1] for m in metrics], w, label="ViT+VQC", color="#7F4FBF")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Comparison: ViT-only vs ViT+VQC (APTOS Test Set)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/model_comparison.png")

    # Cross-dataset comparison chart
    datasets = ["APTOS", "Messidor", "IDRiD", "EyePACS"]
    vit_g1  = [0.732, 0.033, 0.318, 0.181]
    vqc_g1  = [0.786, 0.056, 0.500, 0.190]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, vit_g1, w, label="ViT-only", color="#4A90D9")
    ax.bar(x + w/2, vqc_g1, w, label="ViT+VQC", color="#7F4FBF")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Grade-1 Sensitivity")
    ax.set_title("Early Detection (G1-Sensitivity) Across Datasets")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/crossdataset_g1_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/crossdataset_g1_comparison.png")

    # Ablation table chart
    models = ["VQC only\n(Config B)", "ViT only\n(Config A)", "ViT+VQC\n(Config C)"]
    acc = [0.546, 0.784, 0.816]
    kap = [0.289, 0.881, 0.872]
    g1s = [0.000, 0.732, 0.786]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, color in zip(axes,
        [acc, kap, g1s],
        ["Accuracy", "Kappa", "G1-Sensitivity"],
        ["#3498DB", "#2ECC71", "#E74C3C"]):
        bars = ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
        ax.set_title(title); ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Ablation Study: Component Contribution", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out}/ablation_chart.png", dpi=150)
    plt.close(fig)
    print(f"  saved {out}/ablation_chart.png")

    print("\nAll visualizations saved to experiments/results/")

if __name__ == "__main__":
    main()
