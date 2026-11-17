"""
Step 3: Statistical significance — 3 seeds, mean ± std.
CHECKPOINT SAVE: saves after every epoch so training can resume if interrupted.
RESUME: detects existing checkpoint and continues from last saved epoch.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import numpy as np
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from torch.optim import AdamW
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

SEEDS  = [42, 123, 456]
EPOCHS = 20
CKPT_DIR = Path("experiments/results/multiseed_checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = Path("experiments/results/multiseed_results.json")

def evaluate(model, loader, device, nc=5):
    model.eval(); probs, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), 1).cpu().numpy()
            probs.append(p); trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

def ckpt_path(seed, model_type, ep=None):
    if ep is None:
        return CKPT_DIR / f"seed{seed}_{model_type}_best.pth"
    return CKPT_DIR / f"seed{seed}_{model_type}_ep{ep:02d}.pth"

def progress_path(seed, model_type):
    return CKPT_DIR / f"seed{seed}_{model_type}_progress.json"

def save_progress(seed, model_type, ep, best_kappa, model_state):
    torch.save(model_state, ckpt_path(seed, model_type))
    meta = {"seed": seed, "model_type": model_type, "last_epoch": ep, "best_kappa": best_kappa}
    with open(progress_path(seed, model_type), "w") as f:
        json.dump(meta, f)
    print(f"  [SAVED] seed{seed} {model_type} ep{ep} kappa={best_kappa:.4f} → {ckpt_path(seed, model_type).name}")

def load_progress(seed, model_type):
    p = progress_path(seed, model_type)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def train_and_eval(model_type, seed, cfg, device):
    set_seed(seed)
    nc = cfg["data"]["num_classes"]
    tr, va, te, info = build_aptos_loaders(cfg, seed=seed, num_workers=0)

    if model_type == "vit":
        model = ViTClassifier(nc).to(device)
        params = [
            {"params": model.backbone.parameters(), "lr": cfg["train"]["lr_backbone"]},
            {"params": model.head.parameters(),     "lr": cfg["train"]["lr_head"]}
        ]
    else:
        model = QuantumRetinaX(cfg).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        params = [{"params": trainable, "lr": cfg["train"]["lr_head"]}]

    criterion = nn.CrossEntropyLoss(weight=info["class_weights"].to(device))
    optim = AdamW(params, weight_decay=cfg["train"]["weight_decay"])

    # Check for existing checkpoint
    meta = load_progress(seed, model_type)
    start_ep   = 1
    best_kappa = -1.0

    if meta and ckpt_path(seed, model_type).exists():
        print(f"  [RESUME] seed{seed} {model_type} — resuming from ep{meta['last_epoch']} (kappa={meta['best_kappa']:.4f})")
        model.load_state_dict(torch.load(ckpt_path(seed, model_type), weights_only=True))
        start_ep   = meta["last_epoch"] + 1
        best_kappa = meta["best_kappa"]
        if start_ep > EPOCHS:
            print(f"  [DONE] seed{seed} {model_type} already completed {EPOCHS} epochs")
            test = evaluate(model, te, device, nc)
            return test

    run_start = time.time()
    for ep in range(start_ep, EPOCHS + 1):
        ep_start = time.time()
        ep_elapsed = ep_start - run_start
        ep_h = int(ep_elapsed // 3600)
        ep_m = int((ep_elapsed % 3600) // 60)
        ep_s = int(ep_elapsed % 60)
        print(f"  [START] ep{ep}/{EPOCHS} at {ep_h:02d}:{ep_m:02d}:{ep_s:02d}", flush=True)

        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            criterion(model(x), y).backward()
            optim.step()

        val = evaluate(model, va, device, nc)

        ep_end = time.time()
        ep_dur = ep_end - ep_start
        total_elapsed = ep_end - run_start
        t_h = int(total_elapsed // 3600)
        t_m = int((total_elapsed % 3600) // 60)
        t_s = int(total_elapsed % 60)
        print(f"  [END]   ep{ep}/{EPOCHS} at {t_h:02d}:{t_m:02d}:{t_s:02d} — took {ep_dur:.1f}s — val_kappa={val['kappa']:.4f}", flush=True)

        if val["kappa"] > best_kappa:
            best_kappa = val["kappa"]

        # Save after EVERY epoch
        save_progress(seed, model_type, ep, best_kappa, model.state_dict())

    # Final test evaluation
    model.load_state_dict(torch.load(ckpt_path(seed, model_type), weights_only=True))
    return evaluate(model, te, device, nc)

def load_existing_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"vit": [], "hybrid": []}

def save_results(results):
    def clean(r):
        return {m: float(v) for m, v in r.items() if m != "per_grade_sensitivity"}
    out = {k: [clean(r) for r in v] for k, v in results.items()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [SAVED] Results → {RESULTS_FILE}")

def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoints saved to: {CKPT_DIR}")
    print(f"Results saved to: {RESULTS_FILE}")

    results = load_existing_results()

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"SEED {seed}")
        print(f"{'='*55}")

        vit_done    = any(r.get("seed") == seed for r in results.get("vit_meta", []))
        hybrid_done = any(r.get("seed") == seed for r in results.get("hybrid_meta", []))

        if seed == 42:
            print("  Using saved weights for seed 42...")
            set_seed(42)
            nc = cfg["data"]["num_classes"]
            _, _, te, _ = build_aptos_loaders(cfg, seed=42, num_workers=0)

            vit = ViTClassifier(nc).to(device)
            vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
            r_vit = evaluate(vit, te, device, nc)
            r_vit["seed"] = 42

            hyb = QuantumRetinaX(cfg).to(device)
            hyb.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
            r_hyb = evaluate(hyb, te, device, nc)
            r_hyb["seed"] = 42

            # Only add if not already there
            if not any(r.get("seed") == 42 for r in results["vit"]):
                results["vit"].append(r_vit)
            if not any(r.get("seed") == 42 for r in results["hybrid"]):
                results["hybrid"].append(r_hyb)

            print(f"  ViT-only : acc={r_vit['accuracy']:.4f}  kappa={r_vit['kappa']:.4f}  g1={r_vit['grade1_sensitivity']:.4f}")
            print(f"  ViT+VQC  : acc={r_hyb['accuracy']:.4f}  kappa={r_hyb['kappa']:.4f}  g1={r_hyb['grade1_sensitivity']:.4f}")
            save_results(results)
            continue

        # ViT-only
        if not any(r.get("seed") == seed for r in results["vit"]):
            print(f"  Training ViT-only  seed {seed}...")
            r_vit = train_and_eval("vit", seed, cfg, device)
            r_vit["seed"] = seed
            results["vit"].append(r_vit)
            save_results(results)
            print(f"  ViT-only : acc={r_vit['accuracy']:.4f}  kappa={r_vit['kappa']:.4f}")
        else:
            print(f"  ViT-only seed {seed} already done — skipping")

        # ViT+VQC
        if not any(r.get("seed") == seed for r in results["hybrid"]):
            print(f"  Training ViT+VQC   seed {seed}...")
            r_hyb = train_and_eval("hybrid", seed, cfg, device)
            r_hyb["seed"] = seed
            results["hybrid"].append(r_hyb)
            save_results(results)
            print(f"  ViT+VQC  : acc={r_hyb['accuracy']:.4f}  kappa={r_hyb['kappa']:.4f}")
        else:
            print(f"  ViT+VQC seed {seed} already done — skipping")

    # Final summary
    print("\n" + "="*60)
    print("FINAL STATISTICAL RESULTS")
    print("="*60)
    for key, label in [("vit", "ViT-only"), ("hybrid", "ViT+VQC")]:
        if len(results[key]) < 2:
            print(f"{label}: only {len(results[key])} seed(s) done so far")
            continue
        accs   = [r["accuracy"]           for r in results[key]]
        kappas = [r["kappa"]              for r in results[key]]
        g1s    = [r["grade1_sensitivity"] for r in results[key]]
        print(f"\n{label} ({len(accs)} seeds):")
        print(f"  Accuracy : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
        print(f"  Kappa    : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        print(f"  G1-Sens  : {np.mean(g1s)*100:.2f} ± {np.std(g1s)*100:.2f}%")
        print(f"  Per-seed : {[round(a*100,1) for a in accs]}")
    print("="*60)

if __name__ == "__main__":
    main()

