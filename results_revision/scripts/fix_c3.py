"""
Task 2: C3 runtime redo.
- torch.cuda.synchronize() before/after each timed region
- warm-up=10, repeats>=50
- peak GPU memory: load ONE model at a time, reset_peak_memory_stats before
- device name and diff_method read from model code (not hard-coded)
- ONE real training epoch on 100-sample low-data set for both models
"""
import sys, csv, time, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX

cfg = load_config()
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]

# ── Read VQC simulator name and diff_method from model code ────────────
import inspect, re
from models.quantum import QuantumLayer
src = inspect.getsource(QuantumLayer)
dev_match = re.search(r'qml\.device\("([^"]+)"', src)
diff_match = re.search(r'diff_method="([^"]+)"', src)
vqc_device_name = dev_match.group(1) if dev_match else "UNKNOWN"
vqc_diff_method = diff_match.group(1) if diff_match else "UNKNOWN"
print(f"VQC simulator: {vqc_device_name}, diff_method: {vqc_diff_method}")

# ── GPU device name ────────────────────────────────────────────────────
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "CPU"
print(f"Device: {device} | {gpu_name}")

WARMUP = 10
REPEATS = 50

def sync():
    if device == "cuda":
        torch.cuda.synchronize()

def timed_run(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

# ── Dummy input ────────────────────────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224).to(device)

rows = []

# ─────────────────────────────────────────────────────────────────────
# ViT-only: full pipeline timing + peak GPU mem
# ─────────────────────────────────────────────────────────────────────
print("\n--- ViT-only ---")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit.eval()

# Peak GPU mem after loading
if device == "cuda":
    torch.cuda.synchronize()
    vit_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  ViT-only loaded, peak mem: {vit_mem_mb:.1f} MB")
else:
    vit_mem_mb = float("nan")

# Warm up + time full pipeline
with torch.no_grad():
    mu, sd = timed_run(lambda: vit(dummy))
print(f"  Full pipeline: {mu:.4f} +/- {sd:.4f} ms  (n={REPEATS})")
rows.append({"measurement":"full_vit_per_img_ms","mean":round(mu,4),"std":round(sd,4),"unit":"ms","warmup":WARMUP,"repeats":REPEATS,"note":"ViT-only full, batch=1, cuda sync"})

# Head-only timing (reducer + head for ViT = just the head Linear)
cls_dummy = torch.randn(1, 768).to(device)
with torch.no_grad():
    mu_h, sd_h = timed_run(lambda: vit.head(cls_dummy))
print(f"  Head-only: {mu_h:.4f} +/- {sd_h:.4f} ms")
rows.append({"measurement":"head_only_vit_per_img_ms","mean":round(mu_h,4),"std":round(sd_h,4),"unit":"ms","warmup":WARMUP,"repeats":REPEATS,"note":"ViT head Linear only, batch=1"})

del vit
torch.cuda.empty_cache() if device == "cuda" else None

# ─────────────────────────────────────────────────────────────────────
# ViT+VQC: full pipeline timing + peak GPU mem
# ─────────────────────────────────────────────────────────────────────
print("\n--- ViT+VQC ---")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

hybrid = QuantumRetinaX(cfg).to(device)
hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
hybrid.eval()

if device == "cuda":
    torch.cuda.synchronize()
    hyb_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  ViT+VQC loaded, peak mem: {hyb_mem_mb:.1f} MB")
else:
    hyb_mem_mb = float("nan")

REPEATS_HYB = 50  # VQC is slow; 50 still reasonable
with torch.no_grad():
    mu_h2, sd_h2 = timed_run(lambda: hybrid(dummy), warmup=WARMUP, repeats=REPEATS_HYB)
print(f"  Full pipeline: {mu_h2:.4f} +/- {sd_h2:.4f} ms  (n={REPEATS_HYB})")
rows.append({"measurement":"full_hyb_per_img_ms","mean":round(mu_h2,4),"std":round(sd_h2,4),"unit":"ms","warmup":WARMUP,"repeats":REPEATS_HYB,"note":"ViT+VQC full, batch=1, cuda sync"})

# Reducer+VQC+head (head-only for hybrid = all trainable layers)
red_dummy = torch.randn(1, 768).to(device)
with torch.no_grad():
    def hyb_head_fn():
        r = hybrid.reducer(red_dummy)
        q = hybrid.vqc(r)
        return hybrid.head(q)
    mu_hh, sd_hh = timed_run(hyb_head_fn, warmup=WARMUP, repeats=REPEATS_HYB)
print(f"  Reducer+VQC+head: {mu_hh:.4f} +/- {sd_hh:.4f} ms")
rows.append({"measurement":"head_only_hyb_per_img_ms","mean":round(mu_hh,4),"std":round(sd_hh,4),"unit":"ms","warmup":WARMUP,"repeats":REPEATS_HYB,"note":"reducer+VQC+head, batch=1"})

del hybrid
torch.cuda.empty_cache() if device == "cuda" else None

# ─────────────────────────────────────────────────────────────────────
# Memory rows
# ─────────────────────────────────────────────────────────────────────
rows.append({"measurement":"peak_gpu_mem_vit_mb","mean":round(vit_mem_mb,1),"std":"","unit":"MB","warmup":0,"repeats":1,"note":"after loading ViT-only, before inference"})
rows.append({"measurement":"peak_gpu_mem_hyb_mb","mean":round(hyb_mem_mb,1),"std":"","unit":"MB","warmup":0,"repeats":1,"note":"after loading ViT+VQC, before inference"})

# ─────────────────────────────────────────────────────────────────────
# Device/simulator info rows
# ─────────────────────────────────────────────────────────────────────
rows.append({"measurement":"device","mean":gpu_name,"std":"","unit":"","warmup":0,"repeats":0,"note":"torch.cuda.get_device_name(0)"})
rows.append({"measurement":"vqc_simulator","mean":f"{vqc_device_name} (PennyLane)","std":"","unit":"","warmup":0,"repeats":0,"note":f"diff_method={vqc_diff_method}, CPU; read from models/quantum.py"})

# ─────────────────────────────────────────────────────────────────────
# ONE real training epoch on 100-sample low-data set (both models)
# ─────────────────────────────────────────────────────────────────────
print("\n--- Training epoch timing on 100-sample low-data set ---")
tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)
train_labels = np.array([y for _, y in tr.dataset])
idx_100, _ = train_test_split(list(range(len(train_labels))), train_size=100,
                               stratify=train_labels, random_state=42)
subset = Subset(tr.dataset, idx_100)
sub_loader = DataLoader(subset, batch_size=cfg["train"]["batch_size"],
                        shuffle=True, drop_last=False, num_workers=0)

labels_100 = train_labels[idx_100]
c = Counter(labels_100.tolist())
total_c = sum(c.values())
cw = torch.tensor([total_c/(nc*c.get(k,1)) for k in range(nc)], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=cw)

def run_one_epoch(model, loader, dev):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sync()
    t0 = time.perf_counter()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    sync()
    return (time.perf_counter() - t0)

# ViT epoch
vit2 = ViTClassifier(nc).to(device)
vit2.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
epoch_vit_s = run_one_epoch(vit2, sub_loader, device)
print(f"  ViT-only 1 epoch (100 samples): {epoch_vit_s:.2f}s")
rows.append({"measurement":"epoch_vit_100samples_s","mean":round(epoch_vit_s,2),"std":"","unit":"s","warmup":0,"repeats":1,"note":"1 full training epoch, 100-sample low-data loader"})
del vit2; torch.cuda.empty_cache() if device == "cuda" else None

# Hybrid epoch
hyb2 = QuantumRetinaX(cfg).to(device)
hyb2.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
epoch_hyb_s = run_one_epoch(hyb2, sub_loader, device)
print(f"  ViT+VQC 1 epoch (100 samples): {epoch_hyb_s:.2f}s")
rows.append({"measurement":"epoch_hyb_100samples_s","mean":round(epoch_hyb_s,2),"std":"","unit":"s","warmup":0,"repeats":1,"note":"1 full training epoch, 100-sample low-data loader"})
del hyb2; torch.cuda.empty_cache() if device == "cuda" else None

# ─────────────────────────────────────────────────────────────────────
# Write C_runtime.csv
# ─────────────────────────────────────────────────────────────────────
fields = ["measurement","mean","std","unit","warmup","repeats","note"]
with open("results_revision/data/C_runtime.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("\nSaved: results_revision/C_runtime.csv")
print("Task C3 COMPLETE")
