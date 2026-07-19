import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.config import load_config
from utils.seed import set_seed
from data.transforms import build_transforms
from models.backbone import ViTClassifier

def reshape_transform(tensor, height=14, width=14):
    """Reshape ViT output from (batch, 197, 768) to (batch, 768, 14, 14)
    so GradCAM can process it like a CNN feature map."""
    result = tensor[:, 1:, :]  # remove CLS token -> (batch, 196, 768)
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)  # -> (batch, 768, 14, 14)
    return result

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = cfg["data"]["num_classes"]

    model = ViTClassifier(nc).to(device)
    model.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
    model.eval()

    target_layer = model.backbone.encoder.layer[-1].layernorm_before

    root = Path("data/archive")
    df = pd.read_csv(root / "train.csv")
    transform = build_transforms(cfg, train=False)

    cam = GradCAM(model=model, target_layers=[target_layer],
                  reshape_transform=reshape_transform)

    out = "experiments/results/gradcam"
    Path(out).mkdir(parents=True, exist_ok=True)

    samples = []
    for grade in [0, 1, 2, 3, 4]:
        rows = df[df["diagnosis"] == grade].sample(n=2, random_state=42)
        for _, row in rows.iterrows():
            matches = list(root.rglob(f"{row['id_code']}.*"))
            if matches:
                samples.append((matches[0], grade))

    grade_labels = ["Grade 0\nHealthy", "Grade 1\nMild (Early)", "Grade 2\nModerate",
                     "Grade 3\nSevere", "Grade 4\nProliferative"]

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for i, (path, grade) in enumerate(samples):
        col = grade
        row_offset = (i % 2) * 2
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        tensor = transform(img_pil).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=tensor)[0]
        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
        axes[row_offset, col].imshow(img_np)
        if row_offset == 0:
            axes[row_offset, col].set_title(grade_labels[grade], fontsize=11, fontweight="bold")
        axes[row_offset, col].axis("off")
        axes[row_offset + 1, col].imshow(overlay)
        axes[row_offset + 1, col].axis("off")

    fig.text(0.02, 0.75, "Original", va="center", rotation="vertical", fontsize=13)
    fig.text(0.02, 0.25, "Grad-CAM", va="center", rotation="vertical", fontsize=13)
    fig.suptitle("QuantumRetina-X: Grad-CAM Heatmaps Across DR Grades", fontsize=15)
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(f"{out}/gradcam_grid.png", dpi=150)
    plt.close(fig)
    print(f"[ok] saved {out}/gradcam_grid.png")

    for path, grade in samples[:5]:
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        img_np = np.array(img_resized) / 255.0
        tensor = transform(img_pil).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=tensor)[0]
        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(img_np); ax1.set_title("Original"); ax1.axis("off")
        ax2.imshow(overlay); ax2.set_title("Grad-CAM"); ax2.axis("off")
        label = ["Healthy", "Early Detection Target", "Moderate", "Severe", "Proliferative"][grade]
        fig2.suptitle(f"Grade {grade} - {label}")
        fig2.tight_layout()
        fig2.savefig(f"{out}/gradcam_grade{grade}.png", dpi=150)
        plt.close(fig2)
        print(f"[ok] saved gradcam_grade{grade}.png")

    print("\nGrad-CAM complete. All outputs saved to experiments/results/gradcam/")

if __name__ == "__main__":
    main()
