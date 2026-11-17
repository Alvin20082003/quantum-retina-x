"""Phase 0: Reproduce main APTOS results from saved weights."""
import sys, json
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics
from sklearn.metrics import confusion_matrix

TOLERANCE = 0.3

def evaluate(model, loader, device, nc=5):
    model.eval(); probs, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), 1).cpu().numpy()
            probs.append(p); trues.extend(y.numpy())
    probs = np.concatenate(probs); trues = np.array(trues)
    metrics = compute_metrics(trues, probs, nc)
    preds = probs.argmax(1)
    cm = confusion_matrix(trues, preds, labels=list(range(nc)))
    return metrics, probs, trues, preds, cm

def check(name, key, got, expected):
    diff = abs(got - expected)
    status = "OK" if diff <= TOLERANCE else "FAIL"
    print(f"  {status} {name} {key}: got={got:.4f} expected={expected:.4f} diff={diff:.4f}")
    return status == "OK"

cfg = load_config(); set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]
print(f"\nDevice: {device}")

_, _, te, _ = build_aptos_loaders(cfg, num_workers=0)
trues_all = np.array([y for _,y in te.dataset])
grade_counts = [(trues_all==g).sum() for g in range(nc)]
print(f"Test grade support: {grade_counts}")
expected_support = [271, 56, 150, 29, 44]
support_ok = list(grade_counts) == expected_support
print(f"Support match: {'OK' if support_ok else 'FAIL'}")

results = {}; all_ok = True

print("\n--- ViT-only seed42 ---")
vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
m, probs_vit, trues, preds_vit, cm_vit = evaluate(vit, te, device, nc)
for k,ev in {"accuracy":0.784,"kappa":0.881,"auc_roc":0.936,"grade1_sensitivity":0.732}.items():
    all_ok = check("ViT-only", k, m[k], ev) and all_ok
results["vit_seed42"] = {k:round(float(m[k]),4) for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]}
results["vit_seed42"]["cm"] = cm_vit.tolist()
results["vit_seed42"]["preds"] = preds_vit.tolist()
results["vit_seed42"]["probs"] = probs_vit.tolist()
results["vit_seed42"]["trues"] = trues.tolist()

print("\n--- ViT+VQC seed42 ---")
hyb = QuantumRetinaX(cfg).to(device)
hyb.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
m2, probs_hyb, _, preds_hyb, cm_hyb = evaluate(hyb, te, device, nc)
for k,ev in {"accuracy":0.816,"kappa":0.871,"auc_roc":0.894,"grade1_sensitivity":0.554}.items():
    all_ok = check("ViT+VQC", k, m2[k], ev) and all_ok
results["hyb_seed42"] = {k:round(float(m2[k]),4) for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]}
results["hyb_seed42"]["cm"] = cm_hyb.tolist()
results["hyb_seed42"]["preds"] = preds_hyb.tolist()
results["hyb_seed42"]["probs"] = probs_hyb.tolist()

print("\n--- Low-data ViT-only ---")
vit_low = ViTClassifier(nc).to(device)
vit_low.load_state_dict(torch.load("weights/lowdata_vit_100.pth", weights_only=True))
m3, _, _, preds_vit_low, cm_vit_low = evaluate(vit_low, te, device, nc)
for k,ev in {"accuracy":0.431,"kappa":0.521}.items():
    all_ok = check("ViT-low", k, m3[k], ev) and all_ok
results["vit_low"] = {k:round(float(m3[k]),4) for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]}
results["vit_low"]["cm"] = cm_vit_low.tolist()
results["vit_low"]["preds"] = preds_vit_low.tolist()

print("\n--- Low-data ViT+VQC ---")
hyb_low = QuantumRetinaX(cfg).to(device)
hyb_low.load_state_dict(torch.load("weights/lowdata_hybrid_100.pth", weights_only=True))
m4, _, _, preds_hyb_low, cm_hyb_low = evaluate(hyb_low, te, device, nc)
for k,ev in {"accuracy":0.598,"kappa":0.761}.items():
    all_ok = check("ViT+VQC-low", k, m4[k], ev) and all_ok
results["hyb_low"] = {k:round(float(m4[k]),4) for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]}
results["hyb_low"]["cm"] = cm_hyb_low.tolist()
results["hyb_low"]["preds"] = preds_hyb_low.tolist()

print("\n" + "="*60)
print("PHASE 0 PASSED" if (all_ok and support_ok) else "PHASE 0 FAILED")
Path("results_revision").mkdir(exist_ok=True)
with open("results_revision/data/phase0_results.json","w") as f:
    json.dump(results, f, indent=2)
print("Saved: results_revision/phase0_results.json")
if not (all_ok and support_ok):
    sys.exit(1)
