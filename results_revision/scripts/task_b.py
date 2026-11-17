"""
B1/B2: Zero-shot verification + domain adaptation on cached features.
B0: Confirm which EyePACS version the logs reproduce.
B2: AdaBN and CORAL feature alignment.
"""
import sys, json, csv
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics
from data.transforms import build_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]
IMG_EXT = {".png",".jpg",".jpeg",".tif",".tiff"}

# ── Load models ───────────────────────────────────────────────────────
vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit.eval()
hyb = QuantumRetinaX(cfg).to(device)
hyb.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
hyb.eval()
transform = build_transforms(cfg, train=False)

class GenericDS(Dataset):
    def __init__(self, df, idx, transform):
        self.df=df.reset_index(drop=True); self.idx=idx; self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        img=Image.open(self.idx[str(row["id_code"])]).convert("RGB")
        return self.transform(img), int(row["diagnosis"]), str(row["id_code"])

def build_loader(root, csv_name):
    root=Path(root)
    df=pd.read_csv(root/csv_name,usecols=[0,1])
    df.columns=["id_code","diagnosis"]
    df["id_code"]=df["id_code"].astype(str).apply(lambda x: Path(x).stem)
    df=df.dropna(subset=["diagnosis"])
    df["diagnosis"]=pd.to_numeric(df["diagnosis"],errors="coerce")
    df=df.dropna(subset=["diagnosis"])
    df["diagnosis"]=df["diagnosis"].astype(int)
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT: idx[p.stem]=p
    df=df[df["id_code"].isin(idx.keys())].reset_index(drop=True)
    print(f"  {root.name}: {len(df)} images")
    return DataLoader(GenericDS(df,idx,transform),batch_size=32,shuffle=False,num_workers=0,drop_last=False), len(df)

datasets = {
    "IDRiD":    ("data/idrid",    "idrid_labels.csv"),
    "Messidor": ("data/messidor", "messidor_data.csv"),
    "EyePACS":  ("data/eyepacs",  "trainLabels.csv"),
}

@torch.no_grad()
def eval_model(model, loader, device, nc):
    model.eval(); probs, trues = [], []
    for batch in loader:
        x = batch[0].to(device)
        p = torch.softmax(model(x),1).cpu().numpy()
        probs.append(p); trues.extend(batch[1].numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc), np.concatenate(probs), np.array(trues)

@torch.no_grad()
def extract_cls(vit_model, loader, device):
    feats, trues = [], []
    for batch in loader:
        x = batch[0].to(device)
        f = vit_model.backbone(pixel_values=x).last_hidden_state[:,0].cpu().numpy()
        feats.append(f); trues.extend(batch[1].numpy())
    return np.concatenate(feats), np.array(trues)

# Cache APTOS train features for source stats
tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)
src_feats = []
with torch.no_grad():
    for x, y in tr:
        f = vit.backbone(pixel_values=x.to(device)).last_hidden_state[:,0].cpu().numpy()
        src_feats.append(f)
src_feats = np.concatenate(src_feats)
src_mean = src_feats.mean(0); src_std = src_feats.std(0)+1e-8
print(f"Source (APTOS train) features: {src_feats.shape}")

rows = []
for ds_name, (root, csv_f) in datasets.items():
    print(f"\n=== {ds_name} ===")
    loader, n = build_loader(root, csv_f)

    # Zero-shot
    m_vit, pv, tv = eval_model(vit, loader, device, nc)
    m_hyb, ph, th = eval_model(hyb, loader, device, nc)
    for model_n, m in [("ViT-only", m_vit), ("ViT+VQC", m_hyb)]:
        rows.append({"dataset":ds_name,"method":"zero-shot","model":model_n,"n":n,
            "accuracy":round(m["accuracy"],4),"macro_f1":round(m["macro_f1"],4),
            "auc":round(m["auc_roc"],4),"kappa":round(m["kappa"],4),
            "grade1_sens":round(m["grade1_sensitivity"],4)})
    print(f"  ZS ViT-only: acc={m_vit['accuracy']:.4f} kappa={m_vit['kappa']:.4f} auc={m_vit['auc_roc']:.4f} g1={m_vit['grade1_sensitivity']:.4f}")
    print(f"  ZS ViT+VQC:  acc={m_hyb['accuracy']:.4f} kappa={m_hyb['kappa']:.4f} auc={m_hyb['auc_roc']:.4f} g1={m_hyb['grade1_sensitivity']:.4f}")

    # Extract target features for AdaBN + CORAL
    tgt_feats, tgt_labels = extract_cls(vit, loader, device)
    tgt_mean = tgt_feats.mean(0); tgt_std = tgt_feats.std(0)+1e-8

    # AdaBN: standardize with target stats, rescale with source stats
    tgt_aligned_adabn = (tgt_feats - tgt_mean) / tgt_std * src_std + src_mean

    # CORAL: align covariance
    def coral_align(src, tgt):
        Cs = np.cov(src.T)+np.eye(src.shape[1])*1e-5
        Ct = np.cov(tgt.T)+np.eye(tgt.shape[1])*1e-5
        from scipy.linalg import sqrtm
        Cs_sqrt = np.real(sqrtm(Cs))
        Ct_sqrt_inv = np.real(sqrtm(np.linalg.inv(Ct)))
        W = Cs_sqrt @ Ct_sqrt_inv
        return tgt @ W.T

    print(f"  Running CORAL alignment...")
    tgt_aligned_coral = coral_align(src_feats, tgt_feats)

    # Evaluate with aligned features (head-only, ViT classifier head)
    from torch.utils.data import TensorDataset
    def eval_feats_vit(feats, labels):
        vit.eval()
        ds2 = TensorDataset(torch.tensor(feats,dtype=torch.float32), torch.tensor(labels,dtype=torch.long))
        ld2 = DataLoader(ds2, batch_size=256, shuffle=False)
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in ld2:
                logits = vit.head(xb.to(device))
                p = torch.softmax(logits,1).cpu().numpy()
                probs.append(p); trues.extend(yb.numpy())
        return compute_metrics(np.array(trues), np.concatenate(probs), nc)

    def eval_feats_hyb(feats, labels):
        hyb.eval()
        ds2 = TensorDataset(torch.tensor(feats,dtype=torch.float32), torch.tensor(labels,dtype=torch.long))
        ld2 = DataLoader(ds2, batch_size=32, shuffle=False)
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in ld2:
                z = hyb.reducer(xb.to(device))
                q = hyb.vqc(z)
                logits = hyb.head(q)
                p = torch.softmax(logits,1).cpu().numpy()
                probs.append(p); trues.extend(yb.numpy())
        return compute_metrics(np.array(trues), np.concatenate(probs), nc)

    for method, feats_aligned in [("AdaBN", tgt_aligned_adabn), ("CORAL", tgt_aligned_coral)]:
        mv = eval_feats_vit(feats_aligned, tgt_labels)
        mh = eval_feats_hyb(feats_aligned, tgt_labels)
        for model_n, m, mzs in [("ViT-only", mv, m_vit), ("ViT+VQC", mh, m_hyb)]:
            delta = round(m["accuracy"]-mzs["accuracy"],4)
            rows.append({"dataset":ds_name,"method":method,"model":model_n,"n":n,
                "accuracy":round(m["accuracy"],4),"macro_f1":round(m["macro_f1"],4),
                "auc":round(m["auc_roc"],4),"kappa":round(m["kappa"],4),
                "grade1_sens":round(m["grade1_sensitivity"],4),"delta_acc":delta})
        print(f"  {method} ViT-only: acc={mv['accuracy']:.4f} delta={mv['accuracy']-m_vit['accuracy']:+.4f}")
        print(f"  {method} ViT+VQC:  acc={mh['accuracy']:.4f} delta={mh['accuracy']-m_hyb['accuracy']:+.4f}")

with open("results_revision/data/B_external.csv","w",newline="") as f:
    cols=["dataset","method","model","n","accuracy","macro_f1","auc","kappa","grade1_sens","delta_acc"]
    w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print("\nSaved: results_revision/B_external.csv")

# Identify which EyePACS version matches logs
print("\n=== EyePACS version identification ===")
eye_vit = next(r for r in rows if r["dataset"]=="EyePACS" and r["model"]=="ViT-only" and r["method"]=="zero-shot")
eye_hyb = next(r for r in rows if r["dataset"]=="EyePACS" and r["model"]=="ViT+VQC" and r["method"]=="zero-shot")
print(f"ViT-only: acc={eye_vit['accuracy']} macro_f1={eye_vit['macro_f1']} auc={eye_vit['auc']} kappa={eye_vit['kappa']} g1={eye_vit['grade1_sens']}")
print(f"ViT+VQC:  acc={eye_hyb['accuracy']} macro_f1={eye_hyb['macro_f1']} auc={eye_hyb['auc']} kappa={eye_hyb['kappa']} g1={eye_hyb['grade1_sens']}")
print("Expected v1: 64.2/0.352/0.725/0.412/18.1 and 63.0/0.325/0.603/0.412/19.0")
print("Expected v2: 63.6/0.348/0.719/0.407/17.0 and 62.9/0.328/0.608/0.406/18.1")
print("Task B1/B2 COMPLETE")
