import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from utils.config import load_config
from utils.seed import set_seed
from data.loaders import build_aptos_loaders

def denorm(t, mean, std):
    mean = torch.tensor(mean).view(3,1,1); std = torch.tensor(std).view(3,1,1)
    return (t*std + mean).clamp(0,1)

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    tr, va, te, info = build_aptos_loaders(cfg, num_workers=0)
    print("split sizes  train/val/test:", info["n_train"], info["n_val"], info["n_test"])
    print("train grade distribution:", info["train_dist"])
    print("class weights:", [round(w,2) for w in info["class_weights"].tolist()])

    imgs, labels = next(iter(tr))
    mean, std = cfg["data"]["imagenet_mean"], cfg["data"]["imagenet_std"]
    n = min(16, imgs.size(0)); cols = 4; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6*rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            im = denorm(imgs[i], mean, std).permute(1,2,0).numpy()
            ax.imshow(im); ax.set_title(f"grade {labels[i].item()}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out = "experiments/results/aptos_sample_grid.png"
    fig.savefig(out, dpi=110)
    print(f"[ok] saved grid -> {out}  (open it and eyeball the retinas)")

if __name__ == "__main__":
    main()
