import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX
from training.metrics import compute_metrics

IMG_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

class GenericDataset(Dataset):
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

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    logger = ExperimentLogger(cfg["paths"]["results_dir"])

    root = Path("data/eyepacs")
    df = pd.read_csv(root / "trainLabels.csv")
    # EyePACS uses 'image' and 'level' columns
    df = df.rename(columns={"image": "id_code", "level": "diagnosis"})
    df["id_code"] = df["id_code"].astype(str)

    # Take stratified 5000-image subset
    from sklearn.model_selection import train_test_split
    df_sub, _ = train_test_split(df, train_size=5000, stratify=df["diagnosis"],
                                  random_state=cfg["seed"])
    df_sub = df_sub.reset_index(drop=True)

    idx = index_images(root)
    before = len(df_sub)
    df_sub = df_sub[df_sub["id_code"].isin(idx.keys())].reset_index(drop=True)
    if len(df_sub) < before:
        print(f"  dropped {before - len(df_sub)} rows with no matching image")

    print(f"  EyePACS subset: {len(df_sub)} images")
    print(f"  grades: {dict(sorted(df_sub['diagnosis'].value_counts().items()))}")

    transform = build_transforms(cfg, train=False)
    ds = GenericDataset(df_sub, idx, transform)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=0)

    # Load models
    vit = ViTClassifier(nc).to(device)
    vit.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    vit.eval()

    hybrid = QuantumRetinaX(cfg).to(device)
    hybrid.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
    hybrid.eval()

    print("=" * 60)
    print("EYEPACS (USA) - CROSS-DATASET TEST")
    print("trained on APTOS (India) -> testing on EyePACS (USA)")
    print("=" * 60)

    for name, model in [("ViT-only", vit), ("ViT+VQC", hybrid)]:
        result = evaluate(model, loader, device, nc)
        print(f"  {name:10s} | acc {result['accuracy']:.3f} "
              f"kappa {result['kappa']:.3f} "
              f"f1 {result['macro_f1']:.3f} "
              f"G1-sens {result['grade1_sensitivity']:.3f} "
              f"| per-grade {[round(s,3) for s in result['per_grade_sensitivity']]}")
        logger.log(
            run_name=f"crossval_eyepacs_{name}",
            dataset="eyepacs", model=name,
            accuracy=round(result["accuracy"], 4),
            macro_f1=round(result["macro_f1"], 4),
            auc_roc=round(result["auc_roc"], 4),
            kappa=round(result["kappa"], 4),
            grade1_sensitivity=round(result["grade1_sensitivity"], 4),
            notes=f"5k subset cross-dataset per_grade={[round(s,3) for s in result['per_grade_sensitivity']]}")

    print("\nLogged. EyePACS cross-dataset done.")

if __name__ == "__main__":
    main()
