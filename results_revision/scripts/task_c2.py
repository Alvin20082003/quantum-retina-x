import sys, json, csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from training.metrics import compute_metrics
from torch.utils.data import TensorDataset, DataLoader

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]

print("Caching ViT features...")
vit = ViTClassifier(nc).to(device)
vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit.eval()

def extract_features(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = vit.backbone(pixel_values=x).last_hidden_state[:,0].cpu().numpy()
            feats.append(out); labels.extend(y.numpy())
    return np.concatenate(feats), np.array(labels)

tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)
feat_tr, y_tr = extract_features(tr)
feat_va, y_va = extract_features(va)
feat_te, y_te = extract_features(te)
print(f"Features cached: train={feat_tr.shape}")

class BaseA(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768,4), nn.Linear(4,nc))
    def forward(self,x): return self.net(x)

class ClassicalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Parameter(torch.ones(4)*0.01)
        self.b1 = nn.Parameter(torch.zeros(4))
        self.a2 = nn.Parameter(torch.ones(4)*0.01)
        self.b2 = nn.Parameter(torch.zeros(4))
    def forward(self,x):
        return torch.tanh(self.a2*torch.tanh(self.a1*x+self.b1)+self.b2)

class BaseB(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.r = nn.Linear(768,4)
        self.cb = ClassicalBlock()
        self.h = nn.Linear(4,nc)
    def forward(self,x): return self.h(self.cb(self.r(x)))

def make_loader(feat, labels, batch=8, shuffle=False):
    ds = TensorDataset(torch.tensor(feat,dtype=torch.float32), torch.tensor(labels,dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

def train_eval(ModelClass, seed, epochs=20):
    set_seed(seed)
    model = ModelClass(nc).to(device)
    cw = info["class_weights"].to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=cw)
    tr_ld = make_loader(feat_tr, y_tr, shuffle=True)
    va_ld = make_loader(feat_va, y_va)
    te_ld = make_loader(feat_te, y_te)
    best_k = -1.0; best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                p = torch.softmax(model(xb.to(device)),1).cpu().numpy()
                probs.append(p); trues.extend(yb.numpy())
        m = compute_metrics(np.array(trues), np.concatenate(probs), nc)
        if m["kappa"] > best_k:
            best_k = m["kappa"]
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = torch.softmax(model(xb.to(device)),1).cpu().numpy()
            probs.append(p); trues.extend(yb.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

seeds = [42, 123, 456]
rows = []
for name, MC in [("Classical-A", BaseA), ("Classical-B", BaseB)]:
    print(f"\nTraining {name}...")
    results = []
    for s in seeds:
        print(f"  seed {s}...", end="", flush=True)
        m = train_eval(MC, s)
        results.append(m)
        print(f" acc={m['accuracy']:.4f} kappa={m['kappa']:.4f}")
    for metric in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]:
        vals = [r[metric] for r in results]
        rows.append({
            "model": name, "metric": metric,
            "seed42": round(vals[0],4), "seed123": round(vals[1],4), "seed456": round(vals[2],4),
            "mean": round(float(np.mean(vals)),4), "std": round(float(np.std(vals,ddof=1)),4)
        })
    print(f"  acc: {np.mean([r['accuracy'] for r in results]):.4f}+-{np.std([r['accuracy'] for r in results],ddof=1):.4f}")

with open("results_revision/data/C_baselines.csv","w",newline="") as f:
    w = csv.DictWriter(f,fieldnames=["model","metric","seed42","seed123","seed456","mean","std"])
    w.writeheader(); w.writerows(rows)
print("\nSaved: results_revision/C_baselines.csv")
print("Task C2 COMPLETE")
