"""
Low-data regime evaluation for Paper 1.
Compares ViT-only vs ViT+VQC trained on only 100 samples.
Weights already saved from previous lowdata experiments.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

def evaluate(model, loader, device, nc=5):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), 1).cpu().numpy()
            probs.append(p)
            trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    print(f"Device: {device}")

    # Use full test set (same 550 samples) for fair comparison
    _, _, te, _ = build_aptos_loaders(cfg, num_workers=0)

    print("\n" + "="*60)
    print("LOW-DATA REGIME EVALUATION (trained on 100 samples)")
    print("="*60)

    # ViT-only trained on 100 samples
    vit_low = ViTClassifier(nc).to(device)
    vit_low.load_state_dict(torch.load("weights/lowdata_vit_100.pth", weights_only=True))
    r_vit = evaluate(vit_low, te, device, nc)
    print(f"\nViT-only (100 samples):")
    print(f"  Accuracy:      {r_vit['accuracy']:.4f}  ({r_vit['accuracy']*100:.1f}%)")
    print(f"  Macro-F1:      {r_vit['macro_f1']:.4f}")
    print(f"  AUC:           {r_vit['auc_roc']:.4f}")
    print(f"  Kappa:         {r_vit['kappa']:.4f}")
    print(f"  Grade-1 Sens:  {r_vit['grade1_sensitivity']:.4f}  ({r_vit['grade1_sensitivity']*100:.1f}%)")

    # ViT+VQC trained on 100 samples
    hybrid_low = QuantumRetinaX(cfg).to(device)
    hybrid_low.load_state_dict(torch.load("weights/lowdata_hybrid_100.pth", weights_only=True))
    r_hyb = evaluate(hybrid_low, te, device, nc)
    print(f"\nViT+VQC (100 samples):")
    print(f"  Accuracy:      {r_hyb['accuracy']:.4f}  ({r_hyb['accuracy']*100:.1f}%)")
    print(f"  Macro-F1:      {r_hyb['macro_f1']:.4f}")
    print(f"  AUC:           {r_hyb['auc_roc']:.4f}")
    print(f"  Kappa:         {r_hyb['kappa']:.4f}")
    print(f"  Grade-1 Sens:  {r_hyb['grade1_sensitivity']:.4f}  ({r_hyb['grade1_sensitivity']*100:.1f}%)")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Setting':<30} {'Accuracy':>9} {'Kappa':>8} {'G1-Sens':>9}")
    print("-"*60)
    print(f"{'ViT-only (full 2563)':<30} {'78.4%':>9} {'0.881':>8} {'73.2%':>9}")
    print(f"{'ViT+VQC (full 2563)':<30} {'81.6%':>9} {'0.871':>8} {'55.4%':>9}")
    print(f"{'ViT-only (100 samples)':<30} {r_vit['accuracy']*100:>8.1f}% {r_vit['kappa']:>8.3f} {r_vit['grade1_sensitivity']*100:>8.1f}%")
    print(f"{'ViT+VQC (100 samples)':<30} {r_hyb['accuracy']*100:>8.1f}% {r_hyb['kappa']:>8.3f} {r_hyb['grade1_sensitivity']*100:>8.1f}%")
    print("="*60)

    # Key finding
    gap_full = 0.8164 - 0.7836
    gap_low  = r_hyb['accuracy'] - r_vit['accuracy']
    print(f"\nAccuracy gap (VQC - ViT):")
    print(f"  Full data:  +{gap_full*100:.1f}%")
    print(f"  100 samples: {gap_low*100:+.1f}%")
    if gap_low > gap_full:
        print("\nâ†’ FINDING: Quantum advantage is LARGER in low-data setting")
    elif gap_low > 0:
        print("\nâ†’ FINDING: Quantum still helps in low-data setting (smaller gain)")
    else:
        print("\nâ†’ FINDING: No quantum advantage in low-data setting")

if __name__ == "__main__":
    main()
