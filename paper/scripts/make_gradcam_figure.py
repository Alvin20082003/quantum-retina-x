"""Fresh Grad-CAM example grid using the current, fixed pipeline (hybrid
model gradients + proportional ellipse fundus masking), for the paper -
the old experiments/results/gradcam/*.png predate today's fixes."""
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backend.main import hybrid_model, cam_hybrid, transform, make_fundus_mask, make_heatmap, device

samples = [
    ("data/idrid/segmentation/DR_Training_Set/Fundus Images/IDRiD_03.jpg", "IDRiD"),
    ("data/idrid/segmentation/DR_Training_Set/Fundus Images/IDRiD_35.jpg", "IDRiD"),
    ("data/messidor/messidor-2/messidor-2/preprocess/20051020_43906_0100_PP.png", "Messidor"),
    ("data/eyepacs/augmented_resized_V2/test/0/0212dd31f623-600.jpg", "EyePACS"),
]

fig, axes = plt.subplots(2, len(samples), figsize=(3.2 * len(samples), 6.4))
for col, (path, dataset) in enumerate(samples):
    img_pil = Image.open(path).convert("RGB")
    img_uint8 = np.array(img_pil.resize((224, 224)))
    img_float = img_uint8.astype(np.float32) / 255.0
    tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(hybrid_model(tensor), 1).cpu().numpy()[0]
    grade = int(np.argmax(probs))
    tensor.requires_grad_(True)
    gray_cam = cam_hybrid(input_tensor=tensor)[0]
    fg_mask = make_fundus_mask(img_uint8)
    heat = make_heatmap(img_float, gray_cam, fg_mask)

    axes[0, col].imshow(img_uint8)
    axes[0, col].set_title(f"{dataset}\nPredicted Grade {grade}", fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(heat)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
fig.suptitle("Grad-CAM on the Hybrid ViT+VQC Model Across Datasets", fontsize=12)
fig.tight_layout()
fig.savefig("paper/fig_gradcam_examples.png", dpi=200)
print("saved paper/fig_gradcam_examples.png")
