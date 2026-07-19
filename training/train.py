import argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    epochs = 1 if args.smoke else (args.epochs or cfg["train"]["epochs"])

    tr, va, te, info = build_aptos_loaders(cfg)
    print("data:", info["n_train"], info["n_val"], info["n_test"],
          "| train dist:", info["train_dist"])
    print("device:", device)

    model = ViTClassifier(nc).to(device)
    criterion = nn.CrossEntropyLoss(weight=info["class_weights"].to(device))
    optim = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
        {"params": model.head.parameters(),     "lr": cfg["train"]["lr_head"]},
    ], weight_decay=cfg["train"]["weight_decay"])

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0; t0 = time.time(); bi = 0
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item()
            if bi % 20 == 0:
                print(f"  ep{ep} batch {bi}/{len(tr)} loss {loss.item():.3f}")
            if args.smoke and bi >= 4:
                break
        val = evaluate(model, va, device, nc)
        print(f"[ep {ep}] loss {running/(bi+1):.3f} | val acc {val['accuracy']:.3f} "
              f"f1 {val['macro_f1']:.3f} kappa {val['kappa']:.3f} "
              f"G1-sens {val['grade1_sensitivity']:.3f} | {time.time()-t0:.0f}s")
        if not args.smoke and val["kappa"] > best:
            best = val["kappa"]
            torch.save(model.state_dict(), "weights/vit_baseline.pth")
            print(f"  saved weights/vit_baseline.pth (kappa {best:.3f})")

    if args.smoke:
        print("SMOKE TEST PASSED - full pipeline runs. Now run the real training.")
        return

    model.load_state_dict(torch.load("weights/vit_baseline.pth"))
    test = evaluate(model, te, device, nc)
    print("TEST:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in test.items()})
    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="vit_baseline", dataset="aptos", model="ViT-Base",
        accuracy=round(test["accuracy"],4), macro_f1=round(test["macro_f1"],4),
        auc_roc=round(test["auc_roc"],4), kappa=round(test["kappa"],4),
        grade1_sensitivity=round(test["grade1_sensitivity"],4),
        notes=f"per_grade_sens={[round(s,3) for s in test['per_grade_sensitivity']]}")
    print("logged to results CSV. Phase 2 (ViT baseline) done.")

if __name__ == "__main__":
    main()
