"""Generates real, non-fabricated pipeline-stage figures for both papers by
calling the actual running backend (/predict) on a real fundus photograph and
saving every image it returns, plus rendering real numeric pipeline values
(preprocessing histograms, reducer/VQC outputs, grade probabilities) as
matplotlib charts. Nothing here is staged or illustrative - every figure is
this project's real model, run on a real image, saved as-is."""
import base64, io, json
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "paper"
IMG_PATH = "data/idrid/Imagenes/Imagenes/IDRiD_001.jpg"
API = "http://localhost:8000/predict?model_type=hybrid"

def save_b64(b64_str, name):
    img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
    path = f"{OUT}/{name}.png"
    img.save(path)
    print(f"  saved {path}  ({img.size[0]}x{img.size[1]})")

def main():
    with open(IMG_PATH, "rb") as f:
        files = {"file": f}
        r = requests.post(API, files=files)
    r.raise_for_status()
    d = r.json()
    with open(f"{OUT}/_pipeline_response.json", "w") as f:
        json.dump(d, f, indent=2)

    print(f"Prediction: {d['grade_label']} ({d['confidence']*100:.1f}%), DME={d['dme_detected']}")
    print("Saving real backend images...")
    save_b64(d["original_b64"], "fig_stage_input")
    save_b64(d["mask_viz_b64"], "fig_stage_fundus_mask")
    save_b64(d["heatmap_b64"], "fig_stage_heatmap")
    save_b64(d["redfree_b64"], "fig_stage_redfree")
    save_b64(d["lesion_map_b64"], "fig_stage_multicolor_merge")
    for concept, b64 in d["concept_heatmaps_b64"].items():
        save_b64(b64, f"fig_stage_concept_{concept}")

    p = d["pipeline"]

    # --- Preprocessing: real raw vs normalized histograms ---
    print("Rendering real preprocessing histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    raw = p["preprocessing"]["raw_hist"]
    norm = p["preprocessing"]["norm_hist"]
    axes[0].bar([b["bin"] for b in raw], [b["count"] for b in raw], width=14, color="#eab308")
    axes[0].set_title(f"Raw pixels (0-255)\nμ={p['preprocessing']['raw_mean']}, σ={p['preprocessing']['raw_std']}", fontsize=9)
    axes[0].set_xlabel("Pixel value", fontsize=8)
    axes[1].bar([b["bin"] for b in norm], [b["count"] for b in norm], width=0.3, color="#a855f7")
    axes[1].set_title(f"Normalized (fed to ViT)\nμ={p['preprocessing']['norm_mean']}, σ={p['preprocessing']['norm_std']}", fontsize=9)
    axes[1].set_xlabel("Normalized value", fontsize=8)
    fig.suptitle("Before / after preprocessing, Eq. 1 applied to this image's real pixels", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_preprocessing_hist.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_preprocessing_hist.png")

    # --- Reducer: real 4 values ---
    print("Rendering real reducer output...")
    fig, ax = plt.subplots(figsize=(5, 2.2))
    vals = p["reducer"]["values"]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals]
    ax.bar([f"x{i}" for i in range(len(vals))], vals, color=colors)
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.set_title(f"Reducer output for this image: 768 → {len(vals)} numbers (Eq. 3)", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_reducer.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_reducer.png")

    # --- VQC: real 4 qubit measurements ---
    print("Rendering real VQC measurement...")
    fig, ax = plt.subplots(figsize=(5, 2.2))
    vals = p["vqc"]["values"]
    ax.bar([f"qubit {i}" for i in range(len(vals))], vals, color="#f59e0b")
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title("4-qubit VQC measurement ⟨Zᵢ⟩ for this image (Eq. 6)", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.05 if v >= 0 else -0.12), f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_vqc.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_vqc.png")

    # --- Head/Output: real grade probabilities ---
    print("Rendering real output probabilities...")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    probs = [x * 100 for x in d["probabilities"]]
    labels = [f"G{i}" for i in range(len(probs))]
    grade_colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444", "#dc2626"]
    bars = ax.bar(labels, probs, color=[grade_colors[i] if i == d["grade"] else "#334155" for i in range(len(probs))])
    ax.set_ylabel("Probability (%)", fontsize=8)
    ax.set_title(f"Real output for this image: {d['grade_label']} ({d['confidence']*100:.1f}%)", fontsize=9)
    for i, v in enumerate(probs):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_output_probs.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_output_probs.png")

    # --- TCAV concept scores, this image ---
    print("Rendering real TCAV concept scores...")
    fig, ax = plt.subplots(figsize=(5, 2.2))
    concepts = list(d["tcav_scores"].keys())
    scores = [d["tcav_scores"][c] * 100 for c in concepts]
    ax.barh(concepts, scores, color="#3b82f6")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Concept score (%)", fontsize=8)
    ax.set_title("Real TCAV concept scores for this image (Eq. 10)", fontsize=9)
    for i, v in enumerate(scores):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_tcav_scores.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_tcav_scores.png")

    # --- DME probability ---
    print("Rendering real DME output...")
    fig, ax = plt.subplots(figsize=(5, 1.6))
    prob = p["dme"]["probability"] * 100
    color = "#ef4444" if d["dme_detected"] else "#10b981"
    ax.barh(["DME probability"], [prob], color=color)
    ax.set_xlim(0, 100)
    ax.axvline(50, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_title(f"Real DME model output: {'Detected' if d['dme_detected'] else 'Not detected'} ({prob:.1f}%, threshold 50%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stage_dme_output.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT}/fig_stage_dme_output.png")

    print("\nAll real pipeline-stage figures saved to paper/. Done.")
    print(f"Kept fraction (fundus mask): {p['fundus_mask']['kept_fraction']}%")
    print(f"Marked fraction (lesion overlay vs input): {d['marked_fraction']}%")

if __name__ == "__main__":
    main()
