import numpy as np
import torch
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics
from utils.logger import ExperimentLogger

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]

    _, _, te, info = build_aptos_loaders(cfg, num_workers=0)
    model = QuantumRetinaX(cfg).to(device)
    model.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    model.eval()

    probs, trues = [], []
    with torch.no_grad():
        for x, y in te:
            x = x.to(device)
            p = torch.softmax(model(x), 1).cpu().numpy()
            probs.append(p); trues.extend(y.numpy())

    test = compute_metrics(np.array(trues), np.concatenate(probs), nc)
    print("TEST:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in test.items()})

    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="hybrid_vit_vqc", dataset="aptos", model="ViT+VQC",
        accuracy=round(test["accuracy"],4), macro_f1=round(test["macro_f1"],4),
        auc_roc=round(test["auc_roc"],4), kappa=round(test["kappa"],4),
        grade1_sensitivity=round(test["grade1_sensitivity"],4),
        notes=f"per_grade_sens={[round(s,3) for s in test['per_grade_sensitivity']]}")
    print("Logged. Phase 4 done.")

if __name__ == "__main__":
    main()
