import argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.loaders import build_dme_loaders
from models.hybrid import QuantumRetinaX

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        probs.append(p); trues.extend(y.numpy())
    probs = np.concatenate(probs); trues = np.array(trues)
    preds = (probs >= 0.5).astype(int)
    sens = recall_score(trues, preds, pos_label=1, zero_division=0)
    spec = recall_score(trues, preds, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(trues, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(trues, preds),
            "f1": f1_score(trues, preds, zero_division=0),
            "auc_roc": auc, "sensitivity": sens, "specificity": spec}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if args.smoke else (args.epochs or cfg["train"]["epochs"])

    tr, va, te, info = build_dme_loaders(cfg)
    print("data:", info["n_train"], info["n_val"], info["n_test"],
          "| train dist:", info["train_dist"], "| pos_weight:", info["pos_weight"].item())
    print("device:", device)

    model = QuantumRetinaX(cfg, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=info["pos_weight"].to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=cfg["train"]["lr_head"],
                  weight_decay=cfg["train"]["weight_decay"])

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0; t0 = time.time(); bi = 0
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward(); optim.step()
            running += loss.item()
            if bi % 20 == 0:
                print(f"  ep{ep} batch {bi}/{len(tr)} loss {loss.item():.3f}")
            if args.smoke and bi >= 4:
                break
        val = evaluate(model, va, device)
        print(f"[ep {ep}] loss {running/(bi+1):.3f} | val acc {val['accuracy']:.3f} "
              f"f1 {val['f1']:.3f} auc {val['auc_roc']:.3f} "
              f"sens {val['sensitivity']:.3f} spec {val['specificity']:.3f} "
              f"| {time.time()-t0:.0f}s")
        if not args.smoke and val["auc_roc"] > best:
            best = val["auc_roc"]
            torch.save(model.state_dict(), "weights/dme_vqc.pth")
            print(f"  saved weights/dme_vqc.pth (auc {best:.3f})")

    if args.smoke:
        print("SMOKE TEST PASSED - DME pipeline runs end-to-end.")
        return

    model.load_state_dict(torch.load("weights/dme_vqc.pth", weights_only=True))
    test = evaluate(model, te, device)
    print("TEST:", {k: round(v, 4) for k, v in test.items()})
    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="dme_vqc", dataset="messidor+idrid", model="ViT+VQC-DME",
        accuracy=round(test["accuracy"], 4), macro_f1=round(test["f1"], 4),
        auc_roc=round(test["auc_roc"], 4),
        notes=f"sensitivity={test['sensitivity']:.3f} specificity={test['specificity']:.3f}")
    print("logged to results CSV. DME detector done.")

if __name__ == "__main__":
    main()
