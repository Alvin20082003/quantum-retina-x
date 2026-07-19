import time, json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

IMG_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

class SimpleDataset(Dataset):
    def __init__(self, df, img_index, transform):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_index[str(row["id_code"])]).convert("RGB")
        return self.transform(img), int(row["diagnosis"])

def index_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(p.stem, p)
    return idx

@torch.no_grad()
def evaluate(model, loader, device, nc=5):
    model.eval(); probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        probs.append(p); trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), nc)

def train_model(model, train_loader, val_loader, cfg, device, nc,
                class_weights, epochs, save_path):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=cfg["train"]["lr_head"],
                  weight_decay=cfg["train"]["weight_decay"])
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0; bi = 0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item()
        val = evaluate(model, val_loader, device, nc)
        if val["kappa"] > best:
            best = val["kappa"]
            torch.save(model.state_dict(), save_path)
        print(f"    ep{ep} loss {running/(bi+1):.3f} kappa {val['kappa']:.3f} "
              f"G1 {val['grade1_sensitivity']:.3f}")
    return best

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    logger = ExperimentLogger(cfg["paths"]["results_dir"])
    out = "experiments/results"
    Path(f"{out}/cross_dataset").mkdir(parents=True, exist_ok=True)

    root = Path("data/eyepacs")
    df = pd.read_csv(root / "trainLabels.csv")
    df = df.rename(columns={"image": "id_code", "level": "diagnosis"})
    df["id_code"] = df["id_code"].astype(str)
    idx = index_images(root)
    df = df[df["id_code"].isin(idx.keys())].reset_index(drop=True)

    df_sub, _ = train_test_split(df, train_size=5000,
                                 stratify=df["diagnosis"], random_state=cfg["seed"])
    train_full, test_df = train_test_split(df_sub, test_size=0.3,
                                           stratify=df_sub["diagnosis"],
                                           random_state=cfg["seed"])
    print(f"EyePACS: {len(train_full)} train pool, {len(test_df)} held-out test")

    tf_train = build_transforms(cfg, train=True)
    tf_test = build_transforms(cfg, train=False)
    bs = cfg["train"]["batch_size"]

    test_loader = DataLoader(SimpleDataset(test_df, idx, tf_test),
                             batch_size=bs, shuffle=False, num_workers=0)

    fractions = [1.0, 0.5, 0.25, 0.1]
    results = {"fraction": [], "n_train": [],
               "vit_acc": [], "vit_kappa": [], "vit_g1": [],
               "hybrid_acc": [], "hybrid_kappa": [], "hybrid_g1": []}

    for frac in fractions:
        if frac < 1.0:
            train_sub, _ = train_test_split(train_full, train_size=frac,
                                            stratify=train_full["diagnosis"],
                                            random_state=cfg["seed"])
        else:
            train_sub = train_full
        n = len(train_sub)
        print(f"\n{'='*55}\nFRACTION {frac*100:.0f}%  -  {n} training images\n{'='*55}")

        c = Counter(train_sub["diagnosis"]); total = sum(c.values())
        cw = torch.tensor([total/(nc*c.get(k,1)) for k in range(nc)], dtype=torch.float)

        train_loader = DataLoader(SimpleDataset(train_sub, idx, tf_train),
                                  batch_size=bs, shuffle=True,
                                  drop_last=True, num_workers=0)
        val_sub = train_sub.sample(frac=0.15, random_state=cfg["seed"])
        val_loader = DataLoader(SimpleDataset(val_sub, idx, tf_test),
                                batch_size=bs, shuffle=False, num_workers=0)

        pct = int(frac*100)

        print(f"  Training ViT-only on {n} images...")
        vit = ViTClassifier(nc).to(device)
        for p in vit.backbone.parameters():
            p.requires_grad = False
        t0 = time.time()
        train_model(vit, train_loader, val_loader, cfg, device, nc, cw,
                    epochs=8, save_path=f"weights/lowdata_vit_{pct}.pth")
        vit.load_state_dict(torch.load(f"weights/lowdata_vit_{pct}.pth", weights_only=True))
        vit_test = evaluate(vit, test_loader, device, nc)
        print(f"  ViT-only  acc {vit_test['accuracy']:.3f} kappa {vit_test['kappa']:.3f} "
              f"G1 {vit_test['grade1_sensitivity']:.3f} | {time.time()-t0:.0f}s")
        del vit; torch.cuda.empty_cache()

        print(f"  Training ViT+VQC on {n} images...")
        hybrid = QuantumRetinaX(cfg).to(device)
        t0 = time.time()
        train_model(hybrid, train_loader, val_loader, cfg, device, nc, cw,
                    epochs=8, save_path=f"weights/lowdata_hybrid_{pct}.pth")
        hybrid.load_state_dict(torch.load(f"weights/lowdata_hybrid_{pct}.pth", weights_only=True))
        hyb_test = evaluate(hybrid, test_loader, device, nc)
        print(f"  ViT+VQC   acc {hyb_test['accuracy']:.3f} kappa {hyb_test['kappa']:.3f} "
              f"G1 {hyb_test['grade1_sensitivity']:.3f} | {time.time()-t0:.0f}s")
        del hybrid; torch.cuda.empty_cache()

        results["fraction"].append(f"{pct}%")
        results["n_train"].append(n)
        results["vit_acc"].append(round(vit_test["accuracy"], 4))
        results["vit_kappa"].append(round(vit_test["kappa"], 4))
        results["vit_g1"].append(round(vit_test["grade1_sensitivity"], 4))
        results["hybrid_acc"].append(round(hyb_test["accuracy"], 4))
        results["hybrid_kappa"].append(round(hyb_test["kappa"], 4))
        results["hybrid_g1"].append(round(hyb_test["grade1_sensitivity"], 4))

        logger.log(run_name=f"lowdata_vit_{pct}pct", dataset="eyepacs", model="ViT-only",
                   accuracy=round(vit_test["accuracy"],4),
                   macro_f1=round(vit_test["macro_f1"],4),
                   auc_roc=round(vit_test["auc_roc"],4),
                   kappa=round(vit_test["kappa"],4),
                   grade1_sensitivity=round(vit_test["grade1_sensitivity"],4),
                   notes=f"low-data {pct}% n={n}")
        logger.log(run_name=f"lowdata_hybrid_{pct}pct", dataset="eyepacs", model="ViT+VQC",
                   accuracy=round(hyb_test["accuracy"],4),
                   macro_f1=round(hyb_test["macro_f1"],4),
                   auc_roc=round(hyb_test["auc_roc"],4),
                   kappa=round(hyb_test["kappa"],4),
                   grade1_sensitivity=round(hyb_test["grade1_sensitivity"],4),
                   notes=f"low-data {pct}% n={n}")

        with open(f"{out}/lowdata_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [checkpoint] results saved after {pct}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = [100, 50, 25, 10]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [("vit_acc","hybrid_acc","Accuracy"),
             ("vit_kappa","hybrid_kappa","Kappa"),
             ("vit_g1","hybrid_g1","G1-Sensitivity")]
    for ax, (vk, hk, title) in zip(axes, pairs):
        ax.plot(fracs, results[vk], "b-o", label="ViT-only", linewidth=2)
        ax.plot(fracs, results[hk], color="purple", marker="s",
                label="ViT+VQC", linewidth=2)
        ax.set_xlabel("Training data (%)"); ax.set_ylabel(title)
        ax.set_title(f"{title} vs data size"); ax.legend()
        ax.invert_xaxis(); ax.grid(alpha=0.3)
    fig.suptitle("Low-data experiment: does quantum help when data is scarce?", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/cross_dataset/lowdata_curve.png", dpi=150)
    plt.close(fig)

    print("\n" + "="*55)
    print("LOW-DATA EXPERIMENT COMPLETE")
    print(pd.DataFrame(results).to_string(index=False))
    print("="*55)

if __name__ == "__main__":
    main()
