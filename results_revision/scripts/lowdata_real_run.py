"""
PART 2: Real 100-sample APTOS low-data experiment.
- 3 seeds: 42, 123, 456
- stratified 100-sample subset from APTOS training split (2563 images)
- Both models use frozen vit_baseline.pth backbone
- ViT-only: new Linear(768->5) head
- ViT+VQC: reducer+VQC+head as in QuantumRetinaX
- 8 epochs, final-epoch weights (no checkpoint selection)
- Original training augmentation from build_transforms(cfg, train=True)
- Class weights from 100-sample subset
- No test-set use during training
- Write seed{S}.json after each seed, then summary.md
"""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

cfg = load_config()
nc  = cfg["data"]["num_classes"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

out = Path("results_revision/lowdata_real")
out.mkdir(parents=True, exist_ok=True)

# ── Fixed test set (same 550 for all seeds) ──────────────────────────
set_seed(cfg["seed"])  # seed=42 for dataset splits
_, _, te, _ = build_aptos_loaders(cfg, num_workers=0)
# Full APTOS training set (2563 images)
tr_full, _, _, _ = build_aptos_loaders(cfg, num_workers=0)
train_labels = np.array([y for _, y in tr_full.dataset])
N_TRAIN = len(train_labels)
print(f"Full APTOS training set: {N_TRAIN}")
print(f"Test set: {len(te.dataset)}")
assert N_TRAIN == 2563, f"Unexpected train size: {N_TRAIN}"
assert len(te.dataset) == 550, f"Unexpected test size: {len(te.dataset)}"

bs      = cfg["train"]["batch_size"]   # 8
lr      = cfg["train"]["lr_head"]      # 1e-3
wd      = cfg["train"]["weight_decay"] # 1e-4
EPOCHS  = 8
SEEDS   = [42, 123, 456]

# ── Transforms ───────────────────────────────────────────────────────
tf_train = build_transforms(cfg, train=True)
tf_test  = build_transforms(cfg, train=False)

# Helper: rebuild training dataset with a different transform
class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base    = base_dataset
        self.indices = indices
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        # base_dataset returns (tensor, label) with test transforms;
        # we need to re-apply train transforms to the RAW image.
        # Access the underlying APTOSDataset directly.
        base_ds = self.base
        idx = self.indices[i]
        row = base_ds.df.iloc[idx]
        from PIL import Image as PILImage
        img = PILImage.open(base_ds.img_index[str(row["id_code"])]).convert("RGB")
        return self.transform(img), int(row["diagnosis"])

# tr_full.dataset is an APTOSDataset
base_ds = tr_full.dataset

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    probs, trues = [], []
    for x, y in loader:
        x = x.to(dev)
        p = torch.softmax(model(x), 1).cpu().numpy()
        probs.append(p)
        trues.extend(y.numpy())
    return np.array(trues), np.concatenate(probs)

def compute_all(trues, probs, nc):
    m = compute_metrics(trues, probs, nc)
    preds = probs.argmax(axis=1)
    cm = [[int((preds[trues==i]==j).sum()) for j in range(nc)] for i in range(nc)]
    prec, rec, f1, sup = precision_recall_fscore_support(
        trues, preds, labels=list(range(nc)), zero_division=0)
    per_class = [{"grade":g, "precision":round(float(prec[g]),4),
                  "recall":round(float(rec[g]),4), "f1":round(float(f1[g]),4),
                  "support":int(sup[g])} for g in range(nc)]
    return m, cm, per_class

def train_one_run(model, subset_ds, class_weights, device):
    """Train for EPOCHS epochs, return per-epoch train_loss and wall time per epoch."""
    loader = DataLoader(subset_ds, batch_size=bs, shuffle=True,
                        drop_last=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=wd)
    epoch_times = []
    epoch_losses = []
    model.train()
    for ep in range(1, EPOCHS+1):
        t0 = time.perf_counter()
        total_loss = 0.0; n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1
        ep_time = time.perf_counter() - t0
        epoch_times.append(round(ep_time, 2))
        epoch_losses.append(round(total_loss/max(n_batches,1), 4))
        print(f"    ep{ep} loss={epoch_losses[-1]:.4f} time={ep_time:.1f}s")
    return epoch_times, epoch_losses

# ── Per-seed accumulator for summary ─────────────────────────────────
all_results = {}

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {SEED}")
    print(f"{'='*60}")
    t_seed = time.time()

    # Draw 100-sample stratified subset
    idx_all = list(range(N_TRAIN))
    idx_100, _ = train_test_split(idx_all, train_size=100,
                                   stratify=train_labels, random_state=SEED)
    labels_100 = train_labels[idx_100]
    dist = Counter(labels_100.tolist())
    assert sum(dist.values()) == 100
    print(f"Subset dist: G0={dist[0]} G1={dist[1]} G2={dist[2]} G3={dist[3]} G4={dist[4]}")

    # Save subset info
    subset_info = {
        "seed": SEED,
        "n": 100,
        "indices": [int(i) for i in idx_100],
        "per_grade_counts": {str(g): int(dist.get(g,0)) for g in range(nc)}
    }
    with open(out/f"A1_real_subset_seed{SEED}.json", "w") as f:
        json.dump(subset_info, f, indent=2)
    print(f"Saved A1_real_subset_seed{SEED}.json")

    # Class weights from 100-sample subset
    total_c = 100
    cw = torch.tensor([total_c/(nc*dist.get(k,1)) for k in range(nc)], dtype=torch.float)
    print(f"Class weights: {[round(float(x),3) for x in cw]}")

    # Train-subset dataset with TRAINING augmentation
    subset_ds = SubsetWithTransform(base_ds, idx_100, tf_train)

    # ── (a) ViT-only: new Linear(768->5) head, backbone from vit_baseline ──
    print(f"\n  [ViT-only]")
    set_seed(SEED)
    vit = ViTClassifier(nc).to(device)
    # Load APTOS-finetuned backbone
    state = torch.load("weights/vit_baseline.pth", weights_only=True)
    # Load backbone weights only, reinit head
    backbone_state = {k: v for k, v in state.items() if k.startswith("backbone.")}
    # New head (random init matching seed)
    head_state = {k: v for k, v in vit.state_dict().items() if k.startswith("head.")}
    vit.load_state_dict({**backbone_state, **head_state})
    # Freeze backbone
    for p in vit.backbone.parameters():
        p.requires_grad = False
    n_trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable}")

    ep_times_vit, ep_losses_vit = train_one_run(vit, subset_ds, cw, device)
    # Evaluate on test set (final-epoch weights)
    trues_te, probs_vit = evaluate(vit, te, device)
    m_vit, cm_vit, pc_vit = compute_all(trues_te, probs_vit, nc)
    print(f"  ViT-only: acc={m_vit['accuracy']:.4f} kappa={m_vit['kappa']:.4f} "
          f"G1-sens={m_vit['grade1_sensitivity']:.4f}")
    del vit; torch.cuda.empty_cache()

    # ── (b) ViT+VQC: backbone from vit_baseline, reducer+VQC+head ────
    print(f"\n  [ViT+VQC]")
    set_seed(SEED)
    hybrid = QuantumRetinaX(cfg).to(device)
    # QuantumRetinaX already loads vit_baseline.pth and freezes backbone
    n_trainable_h = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable_h}")

    ep_times_hyb, ep_losses_hyb = train_one_run(hybrid, subset_ds, cw, device)
    trues_te2, probs_hyb = evaluate(hybrid, te, device)
    assert np.array_equal(trues_te, trues_te2), "Test set order changed between models!"
    m_hyb, cm_hyb, pc_hyb = compute_all(trues_te2, probs_hyb, nc)
    print(f"  ViT+VQC: acc={m_hyb['accuracy']:.4f} kappa={m_hyb['kappa']:.4f} "
          f"G1-sens={m_hyb['grade1_sensitivity']:.4f}")
    del hybrid; torch.cuda.empty_cache()

    # ── Write seed{S}.json ────────────────────────────────────────────
    result = {
        "seed": SEED,
        "training_subset": subset_info["per_grade_counts"],
        "n_train": 100,
        "n_test": 550,
        "epochs": EPOCHS,
        "vit_only": {
            "accuracy":           round(m_vit["accuracy"],4),
            "macro_f1":           round(m_vit["macro_f1"],4),
            "auc_roc":            round(m_vit["auc_roc"],4),
            "kappa":              round(m_vit["kappa"],4),
            "grade1_sensitivity": round(m_vit["grade1_sensitivity"],4),
            "confusion_matrix":   cm_vit,
            "per_class":          pc_vit,
            "epoch_losses":       ep_losses_vit,
            "epoch_times_s":      ep_times_vit,
        },
        "vit_vqc": {
            "accuracy":           round(m_hyb["accuracy"],4),
            "macro_f1":           round(m_hyb["macro_f1"],4),
            "auc_roc":            round(m_hyb["auc_roc"],4),
            "kappa":              round(m_hyb["kappa"],4),
            "grade1_sensitivity": round(m_hyb["grade1_sensitivity"],4),
            "confusion_matrix":   cm_hyb,
            "per_class":          pc_hyb,
            "epoch_losses":       ep_losses_hyb,
            "epoch_times_s":      ep_times_hyb,
        },
        "total_seed_time_s": round(time.time()-t_seed, 1)
    }
    with open(out/f"seed{SEED}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved seed{SEED}.json  (total {result['total_seed_time_s']}s)")
    all_results[SEED] = result

# ── Summary ───────────────────────────────────────────────────────────
completed = list(all_results.keys())
print(f"\n{'='*60}\nCompleted seeds: {completed}\n{'='*60}")

def pop_std(vals):
    a = np.array(vals, dtype=float)
    return float(np.std(a))  # population std

def mean_(vals):
    return float(np.mean(np.array(vals, dtype=float)))

metrics_list = ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]
md_lines = [
    "# Real 100-Sample APTOS Low-Data Experiment",
    "",
    "**Setup:** stratified 100-sample subset of APTOS training split (2563 images), "
    "both models use frozen `weights/vit_baseline.pth` backbone, 8 epochs final-epoch weights, "
    "evaluated on fixed 550-image APTOS test set.",
    "",
    f"**Completed seeds:** {completed}",
    "",
    "## Per-Seed Results",
    "",
    "### ViT-only (new head, frozen backbone)",
    "",
    "| Seed | Acc | Macro-F1 | AUC | Kappa | G1-Sens | G0 | G1 | G2 | G3 | G4 (subset counts) |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for s in completed:
    r = all_results[s]["vit_only"]
    sc = all_results[s]["training_subset"]
    md_lines.append(
        f"| {s} | {r['accuracy']} | {r['macro_f1']} | {r['auc_roc']} | {r['kappa']} "
        f"| {r['grade1_sensitivity']} | {sc['0']} | {sc['1']} | {sc['2']} | {sc['3']} | {sc['4']} |"
    )

md_lines += ["", "### ViT+VQC (reducer+VQC+head, frozen backbone)", "",
             "| Seed | Acc | Macro-F1 | AUC | Kappa | G1-Sens |",
             "|---|---|---|---|---|---|"]
for s in completed:
    r = all_results[s]["vit_vqc"]
    md_lines.append(
        f"| {s} | {r['accuracy']} | {r['macro_f1']} | {r['auc_roc']} | {r['kappa']} "
        f"| {r['grade1_sensitivity']} |"
    )

if len(completed) > 1:
    md_lines += ["", "## Summary Statistics (population std)", ""]
    for model_key, label in [("vit_only","ViT-only"),("vit_vqc","ViT+VQC")]:
        md_lines.append(f"**{label}**")
        md_lines.append("| Metric | " + " | ".join(str(s) for s in completed) + " | Mean | Std |")
        md_lines.append("|---|" + "---|"*(len(completed)+2))
        for m in metrics_list:
            vals = [all_results[s][model_key][m] for s in completed]
            row = f"| {m} | " + " | ".join(f"{v:.4f}" for v in vals)
            row += f" | {mean_(vals):.4f} | {pop_std(vals):.4f} |"
            md_lines.append(row)
        md_lines.append("")
else:
    md_lines += ["", "## Note", f"Only seed {completed[0]} completed. Mean/std not computed."]

with open(out/"summary.md","w") as f:
    f.write("\n".join(md_lines)+"\n")
print("Saved summary.md")

# ── Update run_log.md ─────────────────────────────────────────────────
with open("results_revision/reports/run_log.md") as f:
    rl = f.read()
rl += f"""

## PART 2: Real 100-Sample APTOS Low-Data Experiment

- Completed seeds: {completed}
- Training data: stratified 100 samples from APTOS training split (2563 images)
- Test data: fixed APTOS test split (550 images, same as main experiment)
- Backbone: weights/vit_baseline.pth (APTOS-finetuned from in21k) — FROZEN for both models.
  CAVEAT: this backbone was fine-tuned on the FULL APTOS training set (2563 images).
  The 100-sample training is therefore learning a new head on top of a backbone that has
  already seen APTOS-domain images during full-data training. This is not a truly
  data-scarce scenario from the backbone perspective; it measures head-learning only
  under low-label conditions.
- ViT-only: new Linear(768->5) head initialised with torch default (seed set per run)
- ViT+VQC: reducer+VQC+head from QuantumRetinaX; QuantumRetinaX loads vit_baseline.pth
- 8 epochs, final-epoch weights (no checkpoint selection on validation)
- Class weights computed from 100-sample subset per-grade counts
- Optimizer: AdamW, lr={lr}, weight_decay={wd}, batch_size={bs}
- Results in: results_revision/lowdata_real/
"""
with open("results_revision/reports/run_log.md","w") as f:
    f.write(rl)
print("Updated run_log.md")
print("\nPART 2 COMPLETE")
