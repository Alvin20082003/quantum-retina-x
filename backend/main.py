import io, base64, json, time
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.config import load_config
from utils.seed import set_seed
from data.transforms import build_transforms
from models.backbone import ViTClassifier
from models.hybrid import QuantumRetinaX

app = FastAPI(title="QuantumRetina-X API", version="1.0")
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

transform = build_transforms(cfg, train=False)

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:]
    result = result.reshape(result.size(0), height, width, result.size(2))
    return result.permute(0, 3, 1, 2)

target_layer = vit_model.backbone.encoder.layer[-1].layernorm_after
cam = GradCAM(model=vit_model, target_layers=[target_layer],
              reshape_transform=reshape_transform)

tcav_path = Path("experiments/results/tcav/tcav_scores.json")
tcav_data = json.loads(tcav_path.read_text()) if tcav_path.exists() else {}

GRADE_LABELS = {0:"No DR (Healthy)",1:"Mild NPDR (Early)",2:"Moderate NPDR",3:"Severe NPDR",4:"Proliferative DR"}
RECOMMENDATIONS = {
    0:"No action needed. Routine screening in 12 months.",
    1:"Early signs detected. Schedule follow-up in 6 months. Monitor blood glucose.",
    2:"Moderate disease. Refer to ophthalmologist within 1 month.",
    3:"Severe disease. Immediate ophthalmologist consultation required.",
    4:"Proliferative DR. Urgent specialist referral. Risk of vision loss."
}
RISK_LEVELS = {0:"LOW",1:"MODERATE",2:"HIGH",3:"VERY HIGH",4:"CRITICAL"}

@app.get("/health")
def health():
    return {"status":"ok","device":device,"models_loaded":True}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "hybrid"):
    t0 = time.time()
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized) / 255.0
    tensor = transform(img_pil).unsqueeze(0).to(device)

    if model_type == "vit":
        with torch.no_grad(): logits = vit_model(tensor)
    else:
        with torch.no_grad(): logits = hybrid_model(tensor)

    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    grade = int(np.argmax(probs))
    confidence = float(probs[grade])

    grayscale_cam = cam(input_tensor=tensor)[0]
    overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    buf_orig = io.BytesIO()
    img_resized.save(buf_orig, format="PNG")
    original_b64 = base64.b64encode(buf_orig.getvalue()).decode()

    buf_heat = io.BytesIO()
    Image.fromarray(overlay).save(buf_heat, format="PNG")
    heatmap_b64 = base64.b64encode(buf_heat.getvalue()).decode()

    tcav_scores = {}
    for concept, scores in tcav_data.items():
        tcav_scores[concept] = scores.get(str(grade), 0.0)

    # DME detection derived from edema TCAV score
    edema_score = tcav_scores.get("edema", 0.0)
    dme_detected = edema_score > 0.5
    dme_confidence = round(edema_score * 100, 1)

    # DR severity label
    dr_severity = "NPDR" if grade <= 3 else "PDR"

    early_detection = grade == 1

    # Risk considers both DR and DME
    risk = RISK_LEVELS[grade]
    if dme_detected and risk in ["LOW", "MODERATE"]:
        risk = "HIGH"

    rec = RECOMMENDATIONS[grade]
    if dme_detected:
        rec += " DME detected ? additional macular assessment recommended."

    return {
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in probs],
        "original_b64": original_b64,
        "heatmap_b64": heatmap_b64,
        "tcav_scores": tcav_scores,
        "recommendation": rec,
        "risk_level": risk,
        "early_detection": early_detection,
        "model_type": model_type,
        "inference_time": round(time.time() - t0, 2),
        "image_size": list(img_resized.size),
        "dr_type": dr_severity,
        "dme_detected": dme_detected,
        "dme_confidence": dme_confidence,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
