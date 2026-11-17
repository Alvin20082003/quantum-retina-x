from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .transforms import build_transforms

IMG_EXT = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

def _index_images(root: Path) -> dict:
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(p.stem, p)
    return idx

class APTOSDataset(Dataset):
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

def build_aptos_loaders(cfg, root="data/archive", csv_name="train.csv",
                        val_frac=0.15, test_frac=0.15, seed=None, num_workers=None):
    root = Path(root)
    seed = cfg["seed"] if seed is None else seed
    nw = cfg["train"]["num_workers"] if num_workers is None else num_workers

    df = pd.read_csv(root / csv_name)
    df["id_code"] = df["id_code"].astype(str)
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)

    img_index = _index_images(root)
    before = len(df)
    df = df[df["id_code"].isin(img_index.keys())].reset_index(drop=True)
    if len(df) < before:
        print(f"[loaders] dropped {before-len(df)} rows with no matching image")

    train_df, temp_df = train_test_split(
        df, test_size=val_frac+test_frac, stratify=df["diagnosis"], random_state=seed)
    rel = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel, stratify=temp_df["diagnosis"], random_state=seed)

    bs = cfg["train"]["batch_size"]
    dl = cfg["train"]["drop_last"]
    train_ds = APTOSDataset(train_df, img_index, build_transforms(cfg, True))
    val_ds   = APTOSDataset(val_df,   img_index, build_transforms(cfg, False))
    test_ds  = APTOSDataset(test_df,  img_index, build_transforms(cfg, False))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=dl, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    n_classes = cfg["data"]["num_classes"]
    c = Counter(train_df["diagnosis"]); total = sum(c.values())
    class_weights = torch.tensor(
        [total/(n_classes*c.get(k,1)) for k in range(n_classes)], dtype=torch.float)

    info = {"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "train_dist": dict(sorted(Counter(train_df["diagnosis"]).items())),
            "class_weights": class_weights}
    return train_loader, val_loader, test_loader, info


class DMEDataset(Dataset):
    def __init__(self, df, img_index, transform):
        self.df = df.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_index[str(row["id_code"])]).convert("RGB")
        return self.transform(img), float(row["dme"])


def _load_messidor_dme(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "messidor_data.csv")
    df = df[df["adjudicated_gradable"] == 1]
    df["id_code"] = df["id_code"].apply(lambda x: Path(str(x)).stem)
    df["dme"] = df["adjudicated_dme"].astype(int)
    return df[["id_code", "dme"]]


def _load_idrid_dme(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "idrid_labels.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Risk of macular edema"])
    df["id_code"] = df["id_code"].apply(lambda x: Path(str(x)).stem)
    df["dme"] = (df["Risk of macular edema"].astype(int) >= 1).astype(int)
    return df[["id_code", "dme"]]


def build_dme_loaders(cfg, messidor_root="data/messidor", idrid_root="data/idrid",
                      val_frac=0.15, test_frac=0.15, seed=None, num_workers=None):
    seed = cfg["seed"] if seed is None else seed
    nw = cfg["train"]["num_workers"] if num_workers is None else num_workers

    messidor_root, idrid_root = Path(messidor_root), Path(idrid_root)
    messidor_df = _load_messidor_dme(messidor_root)
    idrid_df = _load_idrid_dme(idrid_root)

    img_index = {**_index_images(messidor_root), **_index_images(idrid_root)}
    df = pd.concat([messidor_df, idrid_df], ignore_index=True)
    before = len(df)
    df = df[df["id_code"].isin(img_index.keys())].reset_index(drop=True)
    if len(df) < before:
        print(f"[loaders] dropped {before-len(df)} DME rows with no matching image")

    train_df, temp_df = train_test_split(
        df, test_size=val_frac+test_frac, stratify=df["dme"], random_state=seed)
    rel = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel, stratify=temp_df["dme"], random_state=seed)

    bs = cfg["train"]["batch_size"]
    dl = cfg["train"]["drop_last"]
    train_ds = DMEDataset(train_df, img_index, build_transforms(cfg, True))
    val_ds   = DMEDataset(val_df,   img_index, build_transforms(cfg, False))
    test_ds  = DMEDataset(test_df,  img_index, build_transforms(cfg, False))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=dl, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    c = Counter(train_df["dme"]); neg, pos = c.get(0, 1), c.get(1, 1)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float)

    info = {"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "train_dist": dict(sorted(Counter(train_df["dme"]).items())),
            "pos_weight": pos_weight}
    return train_loader, val_loader, test_loader, info
