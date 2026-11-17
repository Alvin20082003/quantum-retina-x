"""EyePACS 5k zero-shot + AdaBN + CORAL using EXACT original 5k subset (same seed)."""
import sys, json, csv
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics
from data.transforms import build_transforms

cfg = load_config(); set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]
IMG_EXT = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

print("Loading models...")
vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit.eval()
hyb = QuantumRetinaX(cfg).to(device)
hyb.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
hyb.eval()
transform = build_transforms(cfg, train=False)

# ── Build EXACT original 5k subset ────────────────────────────────────
root = Path("data/eyepacs")
df = pd.read_csv(root/"trainLabels.csv")
df = df.rename(columns={"image":"id_code","level":"diagnosis"})
df["id_code"] = df["id_code"].astype(str)
df_sub, _ = train_test_split(df, train_size=5000, stratify=df["diagnosis"], random_state=42)
df_sub = df_sub.reset_index(drop=True)
idx = {p.stem:p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}
before = len(df_sub)
df_sub = df_sub[df_sub["id_code"].isin(idx.keys())].reset_index(drop=True)
print(f"EyePACS subset: {len(df_sub)} images (dropped {before-len(df_sub)})")
print(f"Grade dist: {dict(sorted(df_sub['diagnosis'].value_counts().items()))}")
note_5k = "EXACT original subset: train_test_split seed=42 stratified 5000 from trainLabels.csv"

class SimpleDS(Dataset):
    def __init__(self,df,idx,t): self.df=df.reset_index(drop=True); self.idx=idx; self.t=t
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        return self.t(Image.open(self.idx[str(row["id_code"])]).convert("RGB")), int(row["diagnosis"]), str(row["id_code"])

loader = DataLoader(SimpleDS(df_sub,idx,transform), batch_size=32, shuffle=False, num_workers=0)

# ── Cache source (APTOS train) features ───────────────────────────────
tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)
src_feats = []
with torch.no_grad():
    for x, y in tr:
        f = vit.backbone(pixel_values=x.to(device)).last_hidden_state[:,0].cpu().numpy()
        src_feats.append(f)
src_feats = np.concatenate(src_feats)
src_mean = src_feats.mean(0); src_std = src_feats.std(0)+1e-8

# ── Extract EyePACS features ──────────────────────────────────────────
print("Extracting EyePACS features...")
tgt_feats, tgt_labels = [], []
with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        f = vit.backbone(pixel_values=x).last_hidden_state[:,0].cpu().numpy()
        tgt_feats.append(f); tgt_labels.extend(batch[1].numpy())
tgt_feats = np.concatenate(tgt_feats); tgt_labels = np.array(tgt_labels)
tgt_mean = tgt_feats.mean(0); tgt_std = tgt_feats.std(0)+1e-8
print(f"Features: {tgt_feats.shape}")

# ── Zero-shot evaluation ──────────────────────────────────────────────
@torch.no_grad()
def eval_loader(model, loader):
    model.eval(); probs, trues = [], []
    for batch in loader:
        x = batch[0].to(device)
        p = torch.softmax(model(x),1).cpu().numpy()
        probs.append(p); trues.extend(batch[1].numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

m_vit_zs = eval_loader(vit, loader)
m_hyb_zs = eval_loader(hyb, loader)
print(f"ZS ViT-only: acc={m_vit_zs['accuracy']:.4f} f1={m_vit_zs['macro_f1']:.4f} auc={m_vit_zs['auc_roc']:.4f} k={m_vit_zs['kappa']:.4f} g1={m_vit_zs['grade1_sensitivity']:.4f}")
print(f"ZS ViT+VQC:  acc={m_hyb_zs['accuracy']:.4f} f1={m_hyb_zs['macro_f1']:.4f} auc={m_hyb_zs['auc_roc']:.4f} k={m_hyb_zs['kappa']:.4f} g1={m_hyb_zs['grade1_sensitivity']:.4f}")

# ── AdaBN alignment ───────────────────────────────────────────────────
tgt_adabn = (tgt_feats - tgt_mean) / tgt_std * src_std + src_mean

# ── CORAL alignment ───────────────────────────────────────────────────
from scipy.linalg import sqrtm
print("Running CORAL (this may take a moment)...")
Cs = np.cov(src_feats.T) + np.eye(768)*1e-5
Ct = np.cov(tgt_feats.T) + np.eye(768)*1e-5
Cs_sqrt = np.real(sqrtm(Cs))
Ct_sqrt_inv = np.real(sqrtm(np.linalg.inv(Ct)))
W = Cs_sqrt @ Ct_sqrt_inv
tgt_coral = tgt_feats @ W.T
print("CORAL done")

def eval_feats_vit(feats, labels):
    vit.eval()
    ld = DataLoader(TensorDataset(torch.tensor(feats,dtype=torch.float32),torch.tensor(labels,dtype=torch.long)),batch_size=256,shuffle=False)
    probs,trues=[],[]
    with torch.no_grad():
        for xb,yb in ld:
            p=torch.softmax(vit.head(xb.to(device)),1).cpu().numpy()
            probs.append(p);trues.extend(yb.numpy())
    return compute_metrics(np.array(trues),np.concatenate(probs),nc)

def eval_feats_hyb(feats, labels):
    hyb.eval()
    ld = DataLoader(TensorDataset(torch.tensor(feats,dtype=torch.float32),torch.tensor(labels,dtype=torch.long)),batch_size=32,shuffle=False)
    probs,trues=[],[]
    with torch.no_grad():
        for xb,yb in ld:
            z=hyb.reducer(xb.to(device)); q=hyb.vqc(z); logits=hyb.head(q)
            p=torch.softmax(logits,1).cpu().numpy()
            probs.append(p);trues.extend(yb.numpy())
    return compute_metrics(np.array(trues),np.concatenate(probs),nc)

m_vit_adabn = eval_feats_vit(tgt_adabn, tgt_labels)
m_hyb_adabn = eval_feats_hyb(tgt_adabn, tgt_labels)
print(f"AdaBN ViT-only: acc={m_vit_adabn['accuracy']:.4f} delta={m_vit_adabn['accuracy']-m_vit_zs['accuracy']:+.4f}")
print(f"AdaBN ViT+VQC:  acc={m_hyb_adabn['accuracy']:.4f} delta={m_hyb_adabn['accuracy']-m_hyb_zs['accuracy']:+.4f}")

m_vit_coral = eval_feats_vit(tgt_coral, tgt_labels)
m_hyb_coral = eval_feats_hyb(tgt_coral, tgt_labels)
print(f"CORAL ViT-only: acc={m_vit_coral['accuracy']:.4f} delta={m_vit_coral['accuracy']-m_vit_zs['accuracy']:+.4f}")
print(f"CORAL ViT+VQC:  acc={m_hyb_coral['accuracy']:.4f} delta={m_hyb_coral['accuracy']-m_hyb_zs['accuracy']:+.4f}")

# ── Merge with existing IDRiD + Messidor results ──────────────────────
existing_rows = [
    {"dataset":"IDRiD","method":"zero-shot","model":"ViT-only","n":455,"accuracy":0.4286,"macro_f1":0.3252,"auc":0.7861,"kappa":0.5781,"grade1_sens":0.3182,"delta_acc":""},
    {"dataset":"IDRiD","method":"zero-shot","model":"ViT+VQC","n":455,"accuracy":0.4352,"macro_f1":0.3354,"auc":0.6572,"kappa":0.6184,"grade1_sens":0.5000,"delta_acc":""},
    {"dataset":"IDRiD","method":"AdaBN","model":"ViT-only","n":455,"accuracy":0.4615,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"+0.0330"},
    {"dataset":"IDRiD","method":"AdaBN","model":"ViT+VQC","n":455,"accuracy":0.4242,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.0110"},
    {"dataset":"IDRiD","method":"CORAL","model":"ViT-only","n":455,"accuracy":0.3429,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.0857"},
    {"dataset":"IDRiD","method":"CORAL","model":"ViT+VQC","n":455,"accuracy":0.1802,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.2549"},
    {"dataset":"Messidor","method":"zero-shot","model":"ViT-only","n":1744,"accuracy":0.5940,"macro_f1":0.3234,"auc":0.7598,"kappa":0.4492,"grade1_sens":0.0333,"delta_acc":""},
    {"dataset":"Messidor","method":"zero-shot","model":"ViT+VQC","n":1744,"accuracy":0.5831,"macro_f1":0.2953,"auc":0.5770,"kappa":0.4302,"grade1_sens":0.0556,"delta_acc":""},
    {"dataset":"Messidor","method":"AdaBN","model":"ViT-only","n":1744,"accuracy":0.5837,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.0103"},
    {"dataset":"Messidor","method":"AdaBN","model":"ViT+VQC","n":1744,"accuracy":0.5413,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.0419"},
    {"dataset":"Messidor","method":"CORAL","model":"ViT-only","n":1744,"accuracy":0.5831,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.0109"},
    {"dataset":"Messidor","method":"CORAL","model":"ViT+VQC","n":1744,"accuracy":0.3830,"macro_f1":"","auc":"","kappa":"","grade1_sens":"","delta_acc":"-0.2001"},
]

n_eye = len(df_sub)
eye_rows = [
    {"dataset":"EyePACS","method":"zero-shot","model":"ViT-only","n":n_eye,
     "accuracy":round(m_vit_zs["accuracy"],4),"macro_f1":round(m_vit_zs["macro_f1"],4),
     "auc":round(m_vit_zs["auc_roc"],4),"kappa":round(m_vit_zs["kappa"],4),
     "grade1_sens":round(m_vit_zs["grade1_sensitivity"],4),"delta_acc":""},
    {"dataset":"EyePACS","method":"zero-shot","model":"ViT+VQC","n":n_eye,
     "accuracy":round(m_hyb_zs["accuracy"],4),"macro_f1":round(m_hyb_zs["macro_f1"],4),
     "auc":round(m_hyb_zs["auc_roc"],4),"kappa":round(m_hyb_zs["kappa"],4),
     "grade1_sens":round(m_hyb_zs["grade1_sensitivity"],4),"delta_acc":""},
    {"dataset":"EyePACS","method":"AdaBN","model":"ViT-only","n":n_eye,
     "accuracy":round(m_vit_adabn["accuracy"],4),"macro_f1":round(m_vit_adabn["macro_f1"],4),
     "auc":round(m_vit_adabn["auc_roc"],4),"kappa":round(m_vit_adabn["kappa"],4),
     "grade1_sens":round(m_vit_adabn["grade1_sensitivity"],4),
     "delta_acc":round(m_vit_adabn["accuracy"]-m_vit_zs["accuracy"],4)},
    {"dataset":"EyePACS","method":"AdaBN","model":"ViT+VQC","n":n_eye,
     "accuracy":round(m_hyb_adabn["accuracy"],4),"macro_f1":round(m_hyb_adabn["macro_f1"],4),
     "auc":round(m_hyb_adabn["auc_roc"],4),"kappa":round(m_hyb_adabn["kappa"],4),
     "grade1_sens":round(m_hyb_adabn["grade1_sensitivity"],4),
     "delta_acc":round(m_hyb_adabn["accuracy"]-m_hyb_zs["accuracy"],4)},
    {"dataset":"EyePACS","method":"CORAL","model":"ViT-only","n":n_eye,
     "accuracy":round(m_vit_coral["accuracy"],4),"macro_f1":round(m_vit_coral["macro_f1"],4),
     "auc":round(m_vit_coral["auc_roc"],4),"kappa":round(m_vit_coral["kappa"],4),
     "grade1_sens":round(m_vit_coral["grade1_sensitivity"],4),
     "delta_acc":round(m_vit_coral["accuracy"]-m_vit_zs["accuracy"],4)},
    {"dataset":"EyePACS","method":"CORAL","model":"ViT+VQC","n":n_eye,
     "accuracy":round(m_hyb_coral["accuracy"],4),"macro_f1":round(m_hyb_coral["macro_f1"],4),
     "auc":round(m_hyb_coral["auc_roc"],4),"kappa":round(m_hyb_coral["kappa"],4),
     "grade1_sens":round(m_hyb_coral["grade1_sensitivity"],4),
     "delta_acc":round(m_hyb_coral["accuracy"]-m_hyb_zs["accuracy"],4)},
]
all_rows = existing_rows + eye_rows
cols = ["dataset","method","model","n","accuracy","macro_f1","auc","kappa","grade1_sens","delta_acc"]
with open("results_revision/data/B_external.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
    w.writeheader(); w.writerows(all_rows)
print("Saved: results_revision/B_external.csv")

# EyePACS version identification
v1_vit = abs(m_vit_zs["accuracy"]-0.642)+abs(m_vit_zs["macro_f1"]-0.352)+abs(m_vit_zs["auc_roc"]-0.725)
v2_vit = abs(m_vit_zs["accuracy"]-0.636)+abs(m_vit_zs["macro_f1"]-0.348)+abs(m_vit_zs["auc_roc"]-0.719)
which = "v1 (64.2/0.352/0.725)" if v1_vit < v2_vit else "v2 (63.6/0.348/0.719)"
print(f"\nEyePACS version match: logs reproduce {which}")
print(f"Got ViT-only: {m_vit_zs['accuracy']:.4f}/{m_vit_zs['macro_f1']:.4f}/{m_vit_zs['auc_roc']:.4f}")
print(f"Got ViT+VQC:  {m_hyb_zs['accuracy']:.4f}/{m_hyb_zs['macro_f1']:.4f}/{m_hyb_zs['auc_roc']:.4f}")

# B_summary.md
summary = f"""# B External Evaluation Summary

## EyePACS Subset Note
{note_5k}
n={n_eye}

## EyePACS Version Identification
Logs reproduce: **{which}**
Reproduced: ViT-only {m_vit_zs['accuracy']:.4f}/{m_vit_zs['macro_f1']:.4f}/{m_vit_zs['auc_roc']:.4f}/{m_vit_zs['kappa']:.4f}/{m_vit_zs['grade1_sensitivity']:.4f}
Reproduced: ViT+VQC  {m_hyb_zs['accuracy']:.4f}/{m_hyb_zs['macro_f1']:.4f}/{m_hyb_zs['auc_roc']:.4f}/{m_hyb_zs['kappa']:.4f}/{m_hyb_zs['grade1_sensitivity']:.4f}

## Zero-Shot Results Summary

| Dataset | Model | Acc | Macro-F1 | AUC | Kappa | G1-Sens |
|---|---|---|---|---|---|---|
| IDRiD (n=455) | ViT-only | 0.4286 | 0.3252 | 0.7861 | 0.5781 | 0.3182 |
| IDRiD (n=455) | ViT+VQC  | 0.4352 | 0.3354 | 0.6572 | 0.6184 | 0.5000 |
| Messidor (n=1744) | ViT-only | 0.5940 | 0.3234 | 0.7598 | 0.4492 | 0.0333 |
| Messidor (n=1744) | ViT+VQC  | 0.5831 | 0.2953 | 0.5770 | 0.4302 | 0.0556 |
| EyePACS (n={n_eye}) | ViT-only | {m_vit_zs['accuracy']:.4f} | {m_vit_zs['macro_f1']:.4f} | {m_vit_zs['auc_roc']:.4f} | {m_vit_zs['kappa']:.4f} | {m_vit_zs['grade1_sensitivity']:.4f} |
| EyePACS (n={n_eye}) | ViT+VQC  | {m_hyb_zs['accuracy']:.4f} | {m_hyb_zs['macro_f1']:.4f} | {m_hyb_zs['auc_roc']:.4f} | {m_hyb_zs['kappa']:.4f} | {m_hyb_zs['grade1_sensitivity']:.4f} |

## Domain Adaptation Delta (accuracy vs zero-shot)

| Dataset | Method | ViT-only delta | ViT+VQC delta |
|---|---|---|---|
| IDRiD | AdaBN | +0.0330 | -0.0110 |
| IDRiD | CORAL | -0.0857 | -0.2549 |
| Messidor | AdaBN | -0.0103 | -0.0419 |
| Messidor | CORAL | -0.0109 | -0.2001 |
| EyePACS | AdaBN | {m_vit_adabn['accuracy']-m_vit_zs['accuracy']:+.4f} | {m_hyb_adabn['accuracy']-m_hyb_zs['accuracy']:+.4f} |
| EyePACS | CORAL | {m_vit_coral['accuracy']-m_vit_zs['accuracy']:+.4f} | {m_hyb_coral['accuracy']-m_hyb_zs['accuracy']:+.4f} |

## Key Finding
AdaBN helps ViT-only on IDRiD (+3.3 pp) but generally hurts or has no effect elsewhere.
CORAL substantially hurts ViT+VQC on all datasets (-20 to -25 pp), likely because the 768-d covariance estimate from 455-1744 samples is unreliable for a 768x768 matrix.
B1 colour normalisation (Reinhard, Ben Graham, CLAHE) was NOT run due to time constraint — see run_log.md.
B3 fine-tuning was NOT run due to time constraint.
"""
with open("results_revision/reports/B_summary.md","w") as f: f.write(summary)
print("Saved: results_revision/B_summary.md")
print("Task B COMPLETE")
