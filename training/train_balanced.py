import argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.loaders import build_aptos_loaders
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval(); probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        probs.append(p); trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), num_classes)

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]

    tr, va, te, info = build_aptos_loaders(cfg)
    model = QuantumRetinaX(cfg).to(device)

    cw = info["class_weights"].clone()
    cw[1] *= 3.0
    print(f"class weights: {[round(w,2) for w in cw.tolist()]}  (Grade-1 x3.0)")

    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=cfg["train"]["lr_head"],
                  weight_decay=cfg["train"]["weight_decay"])

    best_kappa = -1.0
    for ep in range(1, 11):
        model.train(); running = 0.0; t0 = time.time(); bi = 0
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item()
            if bi % 40 == 0:
                print(f"  ep{ep} batch {bi}/{len(tr)} loss {loss.item():.3f}")
        val = evaluate(model, va, device, nc)
        g1 = val["grade1_sensitivity"]
        kap = val["kappa"]
        print(f"[ep {ep}] loss {running/(bi+1):.3f} | acc {val['accuracy']:.3f} "
              f"kappa {kap:.3f} G1-sens {g1:.3f} | {time.time()-t0:.0f}s")
        # Save on best kappa BUT only if G1-sens beats ViT baseline
        if g1 >= 0.72 and kap > best_kappa:
            best_kappa = kap
            torch.save(model.state_dict(), "weights/hybrid_best.pth")
            print(f"  SAVED hybrid_best.pth (kappa {kap:.3f} + G1-sens {g1:.3f})")

    model.load_state_dict(torch.load("weights/hybrid_best.pth", weights_only=True))
    test = evaluate(model, te, device, nc)
    print("\nTEST:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in test.items()})
    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="hybrid_best", dataset="aptos", model="ViT+VQC (balanced)",
        accuracy=round(test["accuracy"],4), macro_f1=round(test["macro_f1"],4),
        auc_roc=round(test["auc_roc"],4), kappa=round(test["kappa"],4),
        grade1_sensitivity=round(test["grade1_sensitivity"],4),
        notes=f"g1boost=3.0 g1_threshold=0.72 per_grade={[round(s,3) for s in test['per_grade_sensitivity']]}")
    print("Logged. Balanced hybrid done.")

if __name__ == "__main__":
    main()
