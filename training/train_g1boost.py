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
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1boost", type=float, default=3.0,
                    help="Extra multiplier on Grade-1 class weight")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    epochs = 1 if args.smoke else args.epochs

    tr, va, te, info = build_aptos_loaders(cfg)
    print("data:", info["n_train"], info["n_val"], info["n_test"])

    model = QuantumRetinaX(cfg).to(device)

    # Boost Grade-1 weight specifically for early detection
    cw = info["class_weights"].clone()
    print(f"original class weights: {[round(w,2) for w in cw.tolist()]}")
    cw[1] *= args.g1boost
    print(f"boosted  class weights: {[round(w,2) for w in cw.tolist()]}  (Grade-1 x{args.g1boost})")

    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params, lr=cfg["train"]["lr_head"],
                  weight_decay=cfg["train"]["weight_decay"])

    best = -1.0; best_g1 = 0.0
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0; t0 = time.time(); bi = 0
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item()
            if bi % 40 == 0:
                print(f"  ep{ep} batch {bi}/{len(tr)} loss {loss.item():.3f}")
            if args.smoke and bi >= 4:
                break
        val = evaluate(model, va, device, nc)
        g1 = val["grade1_sensitivity"]
        kap = val["kappa"]
        print(f"[ep {ep}] loss {running/(bi+1):.3f} | val acc {val['accuracy']:.3f} "
              f"f1 {val['macro_f1']:.3f} kappa {kap:.3f} "
              f"G1-sens {g1:.3f} | {time.time()-t0:.0f}s")
        # Save on BEST G1-SENSITIVITY (not kappa) ? early detection is the goal
        if not args.smoke and g1 > best_g1:
            best_g1 = g1; best = kap
            torch.save(model.state_dict(), "weights/hybrid_g1boost.pth")
            print(f"  saved hybrid_g1boost.pth (G1-sens {best_g1:.3f}, kappa {best:.3f})")

    if args.smoke:
        print("SMOKE TEST PASSED"); return

    model.load_state_dict(torch.load("weights/hybrid_g1boost.pth", weights_only=True))
    test = evaluate(model, te, device, nc)
    print("\nTEST:", {k: (round(v,4) if isinstance(v,float) else v) for k,v in test.items()})
    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="hybrid_g1boost", dataset="aptos", model="ViT+VQC (G1-boost)",
        accuracy=round(test["accuracy"],4), macro_f1=round(test["macro_f1"],4),
        auc_roc=round(test["auc_roc"],4), kappa=round(test["kappa"],4),
        grade1_sensitivity=round(test["grade1_sensitivity"],4),
        notes=f"g1boost={args.g1boost} per_grade={[round(s,3) for s in test['per_grade_sensitivity']]}")
    print("Logged. Retune done.")

if __name__ == "__main__":
    main()
