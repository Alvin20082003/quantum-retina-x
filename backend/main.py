import io, base64, json, time, pickle
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from utils.config import load_config
from utils.seed import set_seed
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX

app = FastAPI(title="QuantumRetina-X API", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

cfg = load_config()
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
nc = cfg["data"]["num_classes"]

vit_model = ViTClassifier(nc).to(device)
vit_model.load_state_dict(torch.load("weights/vit_baseline.pth", weights_only=True))
vit_model.eval()

hybrid_model = QuantumRetinaX(cfg).to(device)
hybrid_model.load_state_dict(torch.load("weights/hybrid_vit_vqc.pth", weights_only=True))
hybrid_model.eval()

dme_model = QuantumRetinaX(cfg, num_classes=1).to(device)
dme_model.load_state_dict(torch.load("weights/dme_vqc.pth", weights_only=True))
dme_model.eval()

transform = build_transforms(cfg, train=False)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, result.size(2))
    return result.permute(0, 3, 1, 2)

cam_vit = GradCAM(model=vit_model,
                  target_layers=[vit_model.backbone.encoder.layer[-1].layernorm_before],
                  reshape_transform=reshape_transform)
cam_hybrid = GradCAM(model=hybrid_model,
                     target_layers=[hybrid_model.backbone.encoder.layer[-1].layernorm_before],
                     reshape_transform=reshape_transform)

tcav_dir = Path("experiments/results/tcav")
tcav_probes = {}
for probe_path in tcav_dir.glob("probe_*.pkl"):
    concept = probe_path.stem.replace("probe_", "")
    with open(probe_path, "rb") as f:
        tcav_probes[concept] = pickle.load(f)

CONCEPT_LABELS = {"microaneurysm": "Microaneurysms", "hemorrhage": "Hemorrhages",
                  "exudate": "Hard Exudates", "edema": "Macular Swelling"}
CONCEPT_COLORS = {"microaneurysm": (142, 68, 173), "hemorrhage": (226, 76, 74),
                  "exudate": (34, 197, 94), "edema": (55, 138, 221)}
concept_target_layer = hybrid_model.backbone.encoder.layer[-1].layernorm_before

GRADE_LABELS = {0:"No DR (Healthy)", 1:"Mild NPDR (Early)", 2:"Moderate NPDR",
                3:"Severe NPDR", 4:"Proliferative DR"}
RECOMMENDATIONS = {
    0:"No action needed. Routine screening in 12 months.",
    1:"Early signs detected. Schedule follow-up in 6 months. Monitor blood glucose.",
    2:"Moderate disease. Refer to ophthalmologist within 1 month.",
    3:"Severe disease. Immediate ophthalmologist consultation required.",
    4:"Proliferative DR. Urgent specialist referral. Risk of vision loss."}
RISK_LEVELS = {0:"LOW", 1:"MODERATE", 2:"HIGH", 3:"VERY HIGH", 4:"CRITICAL"}

def to_b64(arr_or_img):
    buf = io.BytesIO()
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def make_redfree(img_np_uint8):
    """Red-free (green channel) view with CLAHE ? what retinal specialists read.
    Hemorrhages and microaneurysms appear as high-contrast dark spots."""
    green = img_np_uint8[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def make_fundus_mask(img_uint8, shrink_frac=0.18):
    """Binary mask of the actual visible retina, excluding the black
    background/corners and a buffer just inside the rim (where ViT's
    LayerNorm blows up activations on the low-variance background patches,
    producing a false 'hot ring' with no clinical meaning).

    Fits an ellipse to the fundus (not a circle - these images are
    noticeably taller than wide) using Otsu's threshold, which adapts to
    each image's own brightness histogram instead of a fixed cutoff. This
    matters because a fixed absolute erosion (an earlier version of this
    function) cuts away a hugely different *proportion* of real retina
    depending on how large the fundus circle already is in frame - e.g.
    EyePACS images fill ~80% of the frame vs ~55-65% for Messidor/IDRiD,
    so the same pixel erosion was excessive on some datasets and
    insufficient on others. Shrinking the fitted ellipse by a fixed
    *percentage* instead keeps the cut proportional across datasets."""
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, raw = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if n <= 1:
        return raw.astype(bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return component.astype(bool)
    (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
    mask = np.zeros(img_uint8.shape[:2], np.uint8)
    axes = (int(major / 2 * (1 - shrink_frac)), int(minor / 2 * (1 - shrink_frac)))
    cv2.ellipse(mask, (int(cx), int(cy)), axes, angle, 0, 360, 1, -1)
    return mask.astype(bool)

def make_mask_viz(img_float, fg_mask):
    """Visualizes the fundus mask itself: the region actually analyzed (real
    retina) is left untouched, the excluded region (black background plus the
    rim buffer just inside it) is darkened and tinted red, so the mask that
    Grad-CAM and the lesion overlays are constrained to is visible as its own
    pipeline stage instead of only being an invisible internal step."""
    excluded = ~fg_mask
    tint = np.zeros_like(img_float)
    tint[..., 0] = 1.0
    out = img_float.copy()
    out[excluded] = out[excluded] * 0.3
    alpha = (excluded.astype(np.float32) * 0.35)[..., None]
    out = out * (1 - alpha) + tint * alpha
    return np.uint8(255 * np.clip(out, 0, 1))

def make_heatmap(img_float, gray_cam, fg_mask, lo_pct=60, hi_pct=99.5):
    """Percentile-stretched Grad-CAM, constrained to the visible retina.
    Suppresses diffuse background activation so genuine hot spots stand out
    instead of washing to uniform pink, and hard-masks anything outside the
    fundus so nothing is ever marked outside the eye."""
    fg_vals = gray_cam[fg_mask]
    if fg_vals.size:
        lo = np.percentile(fg_vals, lo_pct)
        hi = np.percentile(fg_vals, hi_pct)
    else:
        lo, hi = 0.0, 1.0
    stretched = np.clip((gray_cam - lo) / max(hi - lo, 1e-8), 0, 1)
    stretched = np.where(fg_mask, stretched, 0)
    heat = cv2.applyColorMap(np.uint8(255 * stretched), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # weight the blend by activation so cold regions stay readable
    alpha = (stretched[..., None] * 0.75 + 0.15) * fg_mask[..., None]
    blended = heat * alpha + img_float * (1 - alpha)
    return np.uint8(255 * np.clip(blended, 0, 1))

def compute_concept_cam(tensor, probe, height=14, width=14):
    """Grad-CAM-style spatial localization for a linear TCAV concept probe,
    reusing the same target layer as the class-based Grad-CAM but backpropagating
    the concept probe's own linear decision score (mathematically identical to
    sklearn's predict_proba logit, just recomputed in torch so it's
    differentiable) instead of a DR-grade class logit. This answers 'where does
    the model see THIS specific lesion type', not just 'where did it look'."""
    captured = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        captured["activation"] = out

    handle = concept_target_layer.register_forward_hook(fwd_hook)
    try:
        hybrid_model.zero_grad(set_to_none=True)
        out = hybrid_model.backbone(pixel_values=tensor)
        cls = out.last_hidden_state[:, 0]
        w = torch.tensor(probe.coef_[0], dtype=cls.dtype, device=cls.device)
        b = float(probe.intercept_[0])
        score = (cls * w).sum() + b
        score.backward()

        activation = captured["activation"]
        grad = activation.grad
        act_patches = activation[:, 1:, :]
        grad_patches = grad[:, 1:, :]
        weights = grad_patches.mean(dim=1)
        cam = torch.relu((act_patches * weights.unsqueeze(1)).sum(-1))
        cam = cam.reshape(1, 1, height, width)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)
    finally:
        handle.remove()

def make_multicolor_overlay(img_float, concept_cams, fg_mask, colors, threshold=0.3):
    """Combines several per-concept CAMs into one image, each concept in its
    own distinct color, winner-take-all per pixel so overlapping lesions don't
    smear into a muddy blend. threshold=0.3 (not 0.5) trades some precision
    for recall: each concept_cam is normalized to its own max, so a strict
    0.5 cutoff only marks the single hottest spot per concept and misses
    real lesion area that scored lower only relative to that peak, not in
    absolute terms."""
    concepts = list(concept_cams.keys())
    stacked = np.stack([concept_cams[c] for c in concepts], axis=0)
    max_val = stacked.max(axis=0)
    max_idx = stacked.argmax(axis=0)
    color_arr = np.array([colors[c] for c in concepts], dtype=np.float32) / 255.0
    overlay_color = color_arr[max_idx]
    alpha = np.clip((max_val - threshold) / (1 - threshold), 0, 1) * fg_mask
    alpha = alpha[..., None] * 0.8
    blended = overlay_color * alpha + img_float * (1 - alpha)
    return np.uint8(255 * np.clip(blended, 0, 1))

@app.get("/health")
def health():
    return {"status": "ok", "device": device, "models_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "hybrid"):
    t0 = time.time()
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_uint8 = np.array(img_resized)
    img_float = img_uint8.astype(np.float32) / 255.0

    tensor = transform(img_pil).unsqueeze(0).to(device)

    # Real before/after preprocessing statistics (raw 0-255 pixels vs the
    # ImageNet-normalized tensor actually fed to the model), for the
    # pipeline visualization - computed from THIS image, not illustrative.
    raw_flat = img_uint8.astype(np.float32).flatten()
    norm_flat = tensor.detach().cpu().numpy().flatten()
    raw_hist, raw_edges = np.histogram(raw_flat, bins=16, range=(0, 255))
    norm_hist, norm_edges = np.histogram(norm_flat, bins=16, range=(-2.5, 2.5))
    preprocessing_stats = {
        "raw_mean": round(float(raw_flat.mean()), 2), "raw_std": round(float(raw_flat.std()), 2),
        "norm_mean": round(float(norm_flat.mean()), 3), "norm_std": round(float(norm_flat.std()), 3),
        "raw_hist": [{"bin": round(float(raw_edges[i]), 0), "count": int(raw_hist[i])} for i in range(16)],
        "norm_hist": [{"bin": round(float(norm_edges[i]), 2), "count": int(norm_hist[i])} for i in range(16)],
    }

    model = vit_model if model_type == "vit" else hybrid_model
    cam = cam_vit if model_type == "vit" else cam_hybrid
    with torch.no_grad():
        logits = model(tensor)
        dme_prob = torch.sigmoid(dme_model(tensor)).item()
        cls_feat_t = hybrid_model.backbone(pixel_values=tensor).last_hidden_state[:, 0]
        cls_feat = cls_feat_t.cpu().numpy()
        # Real intermediate activations for the pipeline visualization - the
        # actual 4 numbers the reducer and quantum circuit produced for THIS
        # image, not illustrative placeholders.
        reduced_t = hybrid_model.reducer(cls_feat_t)
        quantum_t = hybrid_model.vqc(reduced_t)
        reduced_vals = reduced_t[0].cpu().numpy()
        quantum_vals = quantum_t[0].cpu().numpy()
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    grade = int(np.argmax(probs))
    confidence = float(probs[grade])
    dme_detected = dme_prob >= 0.5

    # Grad-CAM needs a gradient graph back to this layer; the backbone's own
    # weights are frozen, so the input itself must carry requires_grad for
    # that graph to exist at all (see models/hybrid.py forward()).
    tensor.requires_grad_(True)
    gray_cam = cam(input_tensor=tensor)[0]
    fg_mask = make_fundus_mask(img_uint8)
    mask_viz_img = make_mask_viz(img_float, fg_mask)
    heat_img = make_heatmap(img_float, gray_cam, fg_mask)
    redfree_img = make_redfree(img_uint8)

    # Live per-image TCAV: each concept probe scores THIS image's own
    # features, instead of looking up a static table by predicted grade.
    tcav_scores = {c: float(probe.predict_proba(cls_feat)[0, 1])
                   for c, probe in tcav_probes.items()}

    # Per-lesion-type spatial localization, each concept in its own color.
    concept_cams = {c: compute_concept_cam(tensor, probe) for c, probe in tcav_probes.items()}
    lesion_map_img = make_multicolor_overlay(img_float, concept_cams, fg_mask, CONCEPT_COLORS)
    # Individual per-concept heatmaps, before the 4 are merged into one
    # multi-color image - shows what each concept's own raw spatial map
    # looks like ahead of the winner-take-all merge.
    concept_heatmaps_b64 = {c: to_b64(make_heatmap(img_float, cmap, fg_mask))
                             for c, cmap in concept_cams.items()}
    lesion_detections = {
        CONCEPT_LABELS.get(c, c): {"detected": tcav_scores[c] >= 0.5,
                                    "confidence": round(tcav_scores[c] * 100, 1),
                                    "color": "#%02x%02x%02x" % CONCEPT_COLORS[c]}
        for c in tcav_probes
    }
    # Real input-vs-output diff metric: what fraction of the visible retina
    # actually got marked (same threshold as the overlay itself, 0.3).
    marked_mask = np.max(np.stack(list(concept_cams.values())), axis=0) >= 0.3
    marked_fraction = round(float(np.mean(marked_mask & fg_mask) / max(np.mean(fg_mask), 1e-8)) * 100, 1)

    pipeline = {
        "input": {"shape": "224 x 224 x 3", "label": "Fundus image"},
        "preprocessing": preprocessing_stats,
        "backbone": {"label": "ViT-Base/16 (frozen)", "params": "86.4M",
                     "output_dim": 768, "cls_norm": round(float(np.linalg.norm(cls_feat)), 2)},
        "reducer": {"label": "Linear 768 -> 4", "values": [round(float(v), 3) for v in reduced_vals]},
        "vqc": {"label": "4-qubit VQC, 2 layers", "values": [round(float(v), 3) for v in quantum_vals]},
        "head": {"label": f"Linear 4 -> {len(probs)}", "logits": [round(float(v), 3) for v in probs]},
        "fundus_mask": {"label": "Otsu + ellipse fit, 18% shrink",
                         "kept_fraction": round(float(np.mean(fg_mask)) * 100, 1)},
        "dme": {"label": "2nd ViT+VQC model, 1-output head",
                "probability": round(dme_prob, 4), "threshold": 0.5, "detected": dme_detected},
    }

    return {
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in probs],
        "original_b64": to_b64(img_resized),
        "heatmap_b64": to_b64(heat_img),
        "redfree_b64": to_b64(redfree_img),
        "mask_viz_b64": to_b64(mask_viz_img),
        "lesion_map_b64": to_b64(lesion_map_img),
        "concept_heatmaps_b64": concept_heatmaps_b64,
        "lesion_detections": lesion_detections,
        "marked_fraction": marked_fraction,
        "tcav_scores": tcav_scores,
        "pipeline": pipeline,
        "recommendation": RECOMMENDATIONS[grade],
        "risk_level": RISK_LEVELS[grade],
        "dme_detected": dme_detected,
        "dme_confidence": round(dme_prob * 100, 1) if dme_detected else round((1 - dme_prob) * 100, 1),
        "early_detection": grade == 1,
        "model_type": model_type,
        "inference_time": round(time.time() - t0, 2),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
