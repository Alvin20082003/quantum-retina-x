"""
Task 3: B1 Colour normalisation - test-time only, no retraining.
Variants:
  (i)  Reinhard LAB mean/std matching to APTOS training-set statistics
  (ii) Ben Graham: 4*img - 4*GaussianBlur(img, sigma=10) + 128
  (iii) CLAHE on L channel (clipLimit=2.0, tile 8x8)
Parameters fixed in advance, all variants reported.
Datasets: IDRiD (455) first, Messidor (1744) second, EyePACS 5k if time.
Append to B_external.csv and B_summary.md.
"""
import sys, csv, time, json
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

cfg = load_config()
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Compute APTOS training-set LAB statistics for Reinhard
# ─────────────────────────────────────────────────────────────────────
from data.loaders import build_aptos_loaders
tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)

print("Computing APTOS training LAB statistics...")
l_vals, a_vals, b_vals = [], [], []
for idx in range(min(len(tr.dataset), 500)):   # sample 500 for speed
    img, _ = tr.dataset[idx]
    # img is a tensor [3,224,224] after transforms, [0,1] range
    arr = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_vals.append(lab[:,:,0].mean()); a_vals.append(lab[:,:,1].mean()); b_vals.append(lab[:,:,2].mean())

aptos_lab_mean = np.array([np.mean(l_vals), np.mean(a_vals), np.mean(b_vals)])
aptos_lab_std  = np.array([np.std(l_vals),  np.std(a_vals),  np.std(b_vals)])
print(f"APTOS LAB mean: L={aptos_lab_mean[0]:.2f} A={aptos_lab_mean[1]:.2f} B={aptos_lab_mean[2]:.2f}")
print(f"APTOS LAB std:  L={aptos_lab_std[0]:.2f} A={aptos_lab_std[1]:.2f} B={aptos_lab_std[2]:.2f}")

# ─────────────────────────────────────────────────────────────────────
# Normalisation functions (operate on uint8 numpy RGB arrays)
# ─────────────────────────────────────────────────────────────────────
def reinhard_normalise(img_rgb, src_mean=None, src_std=None):
    """Reinhard LAB mean/std transfer to APTOS stats."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if src_mean is None:
        src_mean = np.array([lab[:,:,c].mean() for c in range(3)])
        src_std  = np.array([lab[:,:,c].std()  for c in range(3)])
    for c in range(3):
        if src_std[c] > 1e-6:
            lab[:,:,c] = (lab[:,:,c] - src_mean[c]) / src_std[c] * aptos_lab_std[c] + aptos_lab_mean[c]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def ben_graham(img_rgb):
    """4*img - 4*GaussianBlur(sigma=10) + 128, clipped to [0,255]."""
    blurred = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=10, sigmaY=10)
    out = 4.0 * img_rgb.astype(np.float32) - 4.0 * blurred.astype(np.float32) + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_normalise(img_rgb):
    """CLAHE on L channel of LAB, clipLimit=2.0, tileGridSize=8x8."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe_obj.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

NORM_VARIANTS = {
    "reinhard": reinhard_normalise,
    "ben_graham": ben_graham,
    "clahe": clahe_normalise,
}

# ─────────────────────────────────────────────────────────────────────
# Dataset that applies a normalisation before the standard transforms
# ─────────────────────────────────────────────────────────────────────
IMG_EXT = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def find_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx[p.stem] = p
    return idx

class NormDataset(Dataset):
    def __init__(self, paths, labels, norm_fn, size=224):
        self.paths = paths
        self.labels = labels
        self.norm_fn = norm_fn
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()
        self.normalise = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        img = np.array(self.resize(Image.fromarray(img)))
        img = self.norm_fn(img)
        t = self.to_tensor(Image.fromarray(img))
        t = self.normalise(t)
        return t, int(self.labels[i])

def identity(img_rgb): return img_rgb   # no-op

@torch.no_grad()
def run_eval(model, loader, dev, nc):
    model.eval()
    probs, trues = [], []
    for x, y in loader:
        x = x.to(dev)
        p = torch.softmax(model(x), 1).cpu().numpy()
        probs.append(p); trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

# ─────────────────────────────────────────────────────────────────────
# Load dataset configs
# ─────────────────────────────────────────────────────────────────────
datasets_cfg = {
    "IDRiD": {
        "csv": "data/idrid/labels.csv",
        "img_root": "data/idrid",
        "id_col": "image_id",
        "label_col": "grade",
        "n_expected": 455
    },
    "Messidor": {
        "csv": "data/messidor/labels.csv",
        "img_root": "data/messidor",
        "id_col": "image_id",
        "label_col": "grade",
        "n_expected": 1744
    }
}

# Also try EyePACS 5k
import pandas as pd
from sklearn.model_selection import train_test_split as tts

datasets_cfg["EyePACS"] = {
    "csv": "data/eyepacs/trainLabels.csv",
    "img_root": "data/eyepacs",
    "id_col": "image",
    "label_col": "level",
    "n_expected": 5000,
    "subset": True
}

# ─────────────────────────────────────────────────────────────────────
# Load models (once each)
# ─────────────────────────────────────────────────────────────────────
vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit.eval()

hybrid = QuantumRetinaX(cfg).to(device)
hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
hybrid.eval()

# ─────────────────────────────────────────────────────────────────────
# Read existing B_external.csv to get zero-shot baselines
# ─────────────────────────────────────────────────────────────────────
baseline = {}
with open("results_revision/data/B_external.csv") as f:
    for row in csv.DictReader(f):
        if row["method"] == "zero-shot":
            key = (row["dataset"], row["model"])
            baseline[key] = float(row["accuracy"])

# ─────────────────────────────────────────────────────────────────────
# Results accumulator
# ─────────────────────────────────────────────────────────────────────
new_rows = []
skipped_log = []
t_global = time.time()
DEADLINE_S = 50 * 60  # 50 minutes

def elapsed(): return time.time() - t_global

def build_loader(paths, labels, norm_fn, bs=32):
    ds = NormDataset(paths, labels, norm_fn)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

for ds_name, dscfg in datasets_cfg.items():
    if elapsed() > DEADLINE_S:
        skipped_log.append(f"{ds_name}: SKIPPED - time limit")
        print(f"  SKIPPING {ds_name}: time limit reached")
        continue

    csv_path = Path(dscfg["csv"])
    img_root = dscfg["img_root"]

    if not csv_path.exists():
        skipped_log.append(f"{ds_name}: SKIPPED - CSV not found at {csv_path}")
        print(f"  SKIPPING {ds_name}: CSV not found")
        continue

    df = pd.read_csv(csv_path)
    df = df.rename(columns={dscfg["id_col"]: "id_code", dscfg["label_col"]: "label"})
    df["id_code"] = df["id_code"].astype(str)

    img_idx = find_images(img_root)
    df = df[df["id_code"].isin(img_idx)].reset_index(drop=True)

    if dscfg.get("subset"):
        # EyePACS 5k exact subset (same seed as cross_eyepacs.py)
        df, _ = tts(df, train_size=5000, stratify=df["label"], random_state=42)
        df = df.reset_index(drop=True)

    paths = [img_idx[r] for r in df["id_code"]]
    labels = df["label"].values
    n = len(paths)
    print(f"\n{'='*55}\n{ds_name}  n={n}\n{'='*55}")

    if n < 10:
        skipped_log.append(f"{ds_name}: SKIPPED - only {n} images matched")
        continue

    for variant_name, norm_fn in NORM_VARIANTS.items():
        if elapsed() > DEADLINE_S:
            skipped_log.append(f"{ds_name}/{variant_name}: SKIPPED - time limit")
            continue

        print(f"  Variant: {variant_name}")
        loader = build_loader(paths, labels, norm_fn)

        for model_name, model in [("ViT-only", vit), ("ViT+VQC", hybrid)]:
            if elapsed() > DEADLINE_S:
                skipped_log.append(f"{ds_name}/{variant_name}/{model_name}: SKIPPED - time")
                continue
            m = run_eval(model, loader, device, nc)
            zs_acc = baseline.get((ds_name, model_name), None)
            delta = round(m["accuracy"] - zs_acc, 4) if zs_acc is not None else ""
            row = {
                "dataset": ds_name,
                "method": f"B1_{variant_name}",
                "model": model_name,
                "n": n,
                "accuracy": round(m["accuracy"],4),
                "macro_f1": round(m["macro_f1"],4),
                "auc": round(m["auc_roc"],4),
                "kappa": round(m["kappa"],4),
                "grade1_sens": round(m["grade1_sensitivity"],4),
                "delta_acc": delta
            }
            new_rows.append(row)
            print(f"    {model_name}: acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
                  f"auc={m['auc_roc']:.4f} kappa={m['kappa']:.4f} "
                  f"g1={m['grade1_sensitivity']:.4f} delta={delta}")

# ─────────────────────────────────────────────────────────────────────
# Append to B_external.csv
# ─────────────────────────────────────────────────────────────────────
fields = ["dataset","method","model","n","accuracy","macro_f1","auc","kappa","grade1_sens","delta_acc"]
existing_rows = []
with open("results_revision/data/B_external.csv") as f:
    reader = csv.DictReader(f)
    existing_fields = reader.fieldnames
    for row in reader:
        existing_rows.append(row)

# Merge field names (existing may have different col names; remap if needed)
with open("results_revision/data/B_external.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in existing_rows:
        # remap existing: auc_roc -> auc, grade1_sensitivity -> grade1_sens
        out = {k: row.get(k, row.get("auc_roc" if k=="auc" else
                                     "grade1_sensitivity" if k=="grade1_sens" else k, ""))
               for k in fields}
        w.writerow(out)
    for row in new_rows:
        w.writerow(row)
print(f"\nAppended {len(new_rows)} rows to results_revision/B_external.csv")

# ─────────────────────────────────────────────────────────────────────
# Update B_summary.md
# ─────────────────────────────────────────────────────────────────────
# Read existing summary
with open("results_revision/reports/B_summary.md") as f:
    existing_md = f.read()

# Build new B1 section
b1_table_header = "\n## B1 Colour Normalisation (test-time, no retraining)\n\n"
b1_table_header += "Parameters fixed: Reinhard (APTOS LAB mean/std target), Ben Graham (4x-4xblur+128, sigma=10), CLAHE (clipLimit=2.0, tile 8x8)\n\n"
b1_table_header += "| Dataset | Variant | Model | N | Acc | F1 | AUC | Kappa | G1-Sens | Delta-Acc |\n"
b1_table_header += "|---|---|---|---|---|---|---|---|---|---|\n"

for row in new_rows:
    b1_table_header += (f"| {row['dataset']} | {row['method']} | {row['model']} | {row['n']} "
                        f"| {row['accuracy']} | {row['macro_f1']} | {row['auc']} "
                        f"| {row['kappa']} | {row['grade1_sens']} | {row['delta_acc']} |\n")

if skipped_log:
    b1_table_header += "\n### B1 Skipped\n"
    for s in skipped_log:
        b1_table_header += f"- {s}\n"

# Replace old B1 section if present, otherwise append
if "## B1 Colour Normalisation" in existing_md:
    import re
    existing_md = re.sub(r"\n## B1 Colour Normalisation.*", "", existing_md, flags=re.DOTALL)

# Also update the note at the bottom
existing_md = existing_md.replace(
    "B1 colour normalisation (Reinhard, Ben Graham, CLAHE) was NOT run due to time constraint",
    "B1 colour normalisation (Reinhard, Ben Graham, CLAHE) completed - see table below"
)

with open("results_revision/reports/B_summary.md","w") as f:
    f.write(existing_md.rstrip() + "\n" + b1_table_header)
print("Updated: results_revision/B_summary.md")

# ─────────────────────────────────────────────────────────────────────
# Update run_log.md
# ─────────────────────────────────────────────────────────────────────
with open("results_revision/reports/run_log.md") as f:
    rl = f.read()

rl = rl.replace(
    "- B1 Colour normalisation (Reinhard, Ben Graham, CLAHE): SKIPPED — time constraint",
    "- B1 Colour normalisation (Reinhard, Ben Graham, CLAHE): COMPLETED"
)
if skipped_log:
    rl += "\n\n## B1 Partial Skips\n"
    for s in skipped_log:
        rl += f"- {s}\n"

with open("results_revision/reports/run_log.md","w") as f:
    f.write(rl)
print("Updated: results_revision/run_log.md")
print(f"\nTotal elapsed: {elapsed():.0f}s")
print("Task B1 COMPLETE")
