"""
Improved ViT training with:
- Last 4 transformer layers unfrozen (partial backbone fine-tuning)
- CLAHE fundus preprocessing
- Cosine LR scheduler
- 30 epochs
- Dropout on head
- Label smoothing
- Saves to weights/vit_improved.pth
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger
from data.loaders import build_aptos_loaders
from models.backbone import ViTClassifier
from training.metrics import compute_metrics


def clahe_transform(img: Image.Image) -> Image.Image:
    """Apply CLAHE to green channel — retinal lesions are most visible here."""
    arr = np.array(img)
    green = arr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(green)
    arr[:, :, 1] = green_eq
    return Image.fromarray(arr)


def build_transforms_improved(cfg, train: bool):
    size = cfg["data"]["image_size"]
    mean = cfg["data"]["imagenet_mean"]
    std  = cfg["data"]["imagenet_std"]

    clahe_aug = transforms.Lambda(clahe_transform)

    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            clahe_aug,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        clahe_aug,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ImprovedViTClassifier(nn.Module):
    """ViT with dropout head and partial backbone unfreeze (last 4 layers)."""
    def __init__(self, num_classes=5, unfreeze_last_n=4,
                 pretrained="google/vit-base-patch16-224-in21k"):
        super().__init__()
        from transformers import ViTModel
        self.backbone = ViTModel.from_pretrained(pretrained)
        hidden = self.backbone.config.hidden_size

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N encoder layers
        total_layers = len(self.backbone.encoder.layer)
        for layer in self.backbone.encoder.layer[total_layers - unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Always unfreeze final layernorm
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True

        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        trainable_bb = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[ImprovedViT] backbone: {frozen/1e6:.1f}M frozen, {trainable_bb/1e6:.1f}M unfrozen (last {unfreeze_last_n} layers)")

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval(); probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        probs.append(p); trues.extend(y.numpy())
    return compute_metrics(np.array(trues), np.concatenate(probs), num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--unfreeze", type=int, default=4,
                    help="Number of last ViT encoder layers to unfreeze")
    args = ap.parse_args()

    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]
    epochs = 1 if args.smoke else args.epochs
    print(f"Device: {device} | Epochs: {epochs} | Unfreeze last {args.unfreeze} layers")

    # Use improved transforms
    from data.loaders import APTOSDataset, _index_images
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from torch.utils.data import DataLoader

    root = Path("data/archive")
    df = pd.read_csv(root / "train.csv")
    df["id_code"] = df["id_code"].astype(str)
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    img_index = _index_images(root)
    df = df[df["id_code"].isin(img_index.keys())].reset_index(drop=True)

    seed = cfg["seed"]
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["diagnosis"], random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=seed)

    tr_t = build_transforms_improved(cfg, True)
    va_t = build_transforms_improved(cfg, False)

    bs = cfg["train"]["batch_size"]
    tr_loader = DataLoader(APTOSDataset(train_df, img_index, tr_t),
                           batch_size=bs, shuffle=True, num_workers=2,
                           drop_last=True, pin_memory=True)
    va_loader = DataLoader(APTOSDataset(val_df, img_index, va_t),
                           batch_size=bs, shuffle=False, num_workers=2)
    te_loader = DataLoader(APTOSDataset(test_df, img_index, va_t),
                           batch_size=bs, shuffle=False, num_workers=2)

    c = Counter(train_df["diagnosis"]); total = sum(c.values())
    class_weights = torch.tensor(
        [total/(nc*c.get(k,1)) for k in range(nc)], dtype=torch.float).to(device)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Class weights: {[round(w.item(),2) for w in class_weights]}")

    model = ImprovedViTClassifier(nc, unfreeze_last_n=args.unfreeze).to(device)

    # Differential learning rates: backbone unfrozen layers get lower LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optim = AdamW([
        {"params": backbone_params, "lr": 1e-5},   # low LR for pretrained layers
        {"params": head_params,     "lr": 1e-3},   # higher LR for new head
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); running = 0.0; t0 = time.time(); bi = 0
        for bi, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optim.step()
            running += loss.item()
            if bi % 20 == 0:
                print(f"  ep{ep} batch {bi}/{len(tr_loader)} loss {loss.item():.3f}")
            if args.smoke and bi >= 4:
                break

        scheduler.step()
        val = evaluate(model, va_loader, device, nc)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[ep {ep}] lr={lr_now:.2e} loss {running/(bi+1):.3f} | "
              f"val acc {val['accuracy']:.3f} f1 {val['macro_f1']:.3f} "
              f"kappa {val['kappa']:.3f} G1-sens {val['grade1_sensitivity']:.3f} | "
              f"{time.time()-t0:.0f}s")

        if not args.smoke and val["kappa"] > best:
            best = val["kappa"]
            torch.save(model.state_dict(), "weights/vit_improved.pth")
            print(f"  saved weights/vit_improved.pth (kappa {best:.3f})")

    if args.smoke:
        print("SMOKE TEST PASSED"); return

    model.load_state_dict(torch.load("weights/vit_improved.pth", weights_only=True))
    test = evaluate(model, te_loader, device, nc)
    print("\nTEST RESULTS:")
    print("=" * 50)
    for k, v in test.items():
        print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")
    print("=" * 50)

    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="vit_improved", dataset="aptos", model="ViT-Improved",
        accuracy=round(test["accuracy"],4), macro_f1=round(test["macro_f1"],4),
        auc_roc=round(test["auc_roc"],4), kappa=round(test["kappa"],4),
        grade1_sensitivity=round(test["grade1_sensitivity"],4),
        notes=f"unfreeze={args.unfreeze}layers+CLAHE+cosine+dropout per_grade={[round(s,3) for s in test['per_grade_sensitivity']]}")
    print("Logged. Improved ViT training done.")


if __name__ == "__main__":
    main()
