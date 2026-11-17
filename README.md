# QuantumRetina-X

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.36.0-6929C4?logo=pennylane)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Research_Only-purple)
![IEEE](https://img.shields.io/badge/IEEE-Track_01_%26_02-00629B?logo=ieee)

> **The worlds first hybrid Vision Transformer + Variational Quantum Circuit system for diabetic retinopathy grading, macular edema detection, and AI explainability - with a clinical-grade web UI and an interactive Eye Digital Twin.**

---

## Table of Contents

1. [What is QuantumRetina-X](#1-what-is-quantumretina-x)
2. [Architecture](#2-architecture)
3. [Key Results](#3-key-results)
4. [Clinical Web UI](#4-clinical-web-ui)
5. [Digital Twin Feature](#5-digital-twin-feature)
6. [What Every File Does](#6-what-every-file-does)
7. [Quick Start - Run in 2 Commands](#7-quick-start---run-in-2-commands)
8. [Full Setup](#8-full-setup)
9. [Train the Models](#9-train-the-models)
10. [Run Evaluations](#10-run-evaluations)
11. [Datasets](#11-datasets)
12. [Research Papers](#12-research-papers)

---

## 1. What is QuantumRetina-X

QuantumRetina-X is a research prototype that combines classical deep learning with quantum computing to screen for diabetic eye disease.

```
What it does in 3 seconds:

  Upload fundus photo
        |
        v
  +------------------+     +------------------+     +------------------+
  |  DR Grade        |     |  DME Detection   |     |  XAI Explanation |
  |  G0 -> G4        |     |  Present/Absent  |     |  Grad-CAM + TCAV |
  |  81.6% accuracy  |     |  89.1% accuracy  |     |  4 lesion types  |
  +------------------+     +------------------+     +------------------+
```

**What makes it novel:**
- First paper to combine ViT + VQC for DR grading
- Only 3,125 trainable parameters (backbone is frozen)
- Dual XAI: Grad-CAM spatial + TCAV concept-level explanations
- Eye Digital Twin for patient education
- Tested on 4 datasets across 3 countries (zero-shot transfer)

---

## 2. Architecture

```
Fundus image (224 x 224 x 3 pixels)
          |
          v
+----------------------------------+
|  ViT-Base/16   FROZEN            |  86.4M params
|  google/vit-base-patch16-224     |  196 patches of 16x16 px
|  fine-tuned on APTOS 2019        |  12 transformer layers
+---------------+------------------+
                |  [CLS] token  768-dim feature
                v
+----------------------------------+
|  Reducer  Linear(768 -> 4)       |  trainable  3,076 params
+---------------+------------------+
                |  4-dim compressed vector
                v
+--------------------------------------------------+
|  4-Qubit VQC   PennyLane default.qubit           |  trainable  24 params
|  RY encoding -> 2 layers RX/RY/RZ + CNOT ring   |  runs on CPU
|  Pauli-Z readout -> 4 expectation values         |  parameter-shift diff
+---------------+----------------------------------+
                |  4-dim quantum output
                v
+----------------------------------+
|  Head  Linear(4 -> 5)            |  trainable  25 params
+---------------+------------------+
                |
                v
  G0 / G1 / G2 / G3 / G4   (5-class DR grade)

  Trainable params total: 3,125
  Frozen params total:   86,400,000+
```

**DME model:** Same architecture, `num_classes=1`, sigmoid output. Trained on Messidor + IDRiD.

---

## 3. Key Results

### APTOS 2019 Test Set (n=550, seed 42)

| Model | Accuracy | Macro-F1 | AUC | Kappa | G1-Sens |
|---|---|---|---|---|---|
| VQC-only (no ViT) | 54.6% | 0.237 | 0.646 | 0.289 | 0.0% |
| ViT-only baseline | 78.4% | 0.638 | 0.936 | 0.881 | 73.2% |
| **ViT+VQC (ours)** | **81.6%** | **0.659** | **0.894** | **0.871** | **55.4%** |

### Statistical Reproducibility (3 seeds: 42, 123, 456)

| Model | Seed 42 | Seed 123 | Seed 456 | Mean +- Std |
|---|---|---|---|---|
| ViT-only acc  | 78.4% | 82.5% | 80.9% | 80.6 +- 1.7% |
| ViT-only k    | 0.881 | 0.895 | 0.894 | 0.890 +- 0.006 |
| ViT+VQC acc   | 81.6% | 91.1% | 89.8% | 87.5 +- 4.2% |
| ViT+VQC k     | 0.871 | 0.940 | 0.936 | 0.916 +- 0.032 |

### DME Detection (n=330, Messidor + IDRiD)

| Metric | Value |
|---|---|
| Accuracy | 89.1% |
| AUC | 0.928 |
| Sensitivity | 78.5% |
| Specificity | 91.7% |

### Cross-Dataset Zero-Shot Transfer

| Dataset | Country | n | ViT-only Acc | ViT+VQC Acc | ViT-only AUC | ViT+VQC AUC |
|---|---|---|---|---|---|---|
| APTOS 2019 | India | 550 | 78.4% | **81.6%** | 0.936 | 0.894 |
| IDRiD | India | 455 | 42.9% | 43.5% | 0.786 | 0.657 |
| Messidor-2 | France | 1,744 | 59.4% | 58.3% | 0.760 | 0.577 |
| EyePACS | USA | 5,000 | 64.2% | 63.0% | 0.724 | 0.603 |

---

## 4. Clinical Web UI

A full-stack clinical interface built with React 19 + FastAPI. Dark purple theme throughout.

```
Sidebar Pages:

  Dashboard       -- Upload zone + mock welcome OR live scan results
  Upload Image    -- Dedicated upload page with format/speed info
  Pipeline        -- Static architecture OR live forward-pass visualiser
  Reports         -- Scan history, download heatmap + text report
  Explainability  -- Grad-CAM + TCAV concept explanations
  Digital Twin    -- Interactive eye layer explorer (see Section 5)
  Model Insights  -- Ablation table, cross-dataset results, training strategies
  Settings        -- Model switch, heatmap opacity
  About           -- Research contributions, architecture summary
```

**Dashboard after scan:**
```
+---------------------------------------------+  +---------------------+
|  Original    |  Grad-CAM     |  Lesion Map |  |  Grade 2 Moderate   |
|  fundus      |  heatmap      |  multicolor |  |  Confidence: 81.6%  |
|  image       |  overlay      |  4 lesions  |  +---------------------+
+---------------------------------------------+  |  DME: Not detected  |
|  Heatmap opacity slider                     |  +---------------------+
+---------------------------------------------+  |  TCAV scores        |
|  Lesion detection panel  |  TCAV bars      |  |  Microaneurysm 61%  |
+---------------------------------------------+  |  Hemorrhage   74%   |
|  Grade probability distribution bar chart   |  +---------------------+
+---------------------------------------------+  |  Clinical reco.     |
                                                  +---------------------+
```

**Pipeline page (after scan) - 4 interactive branches:**
```
Forward Pass:
  Input -> Preprocessing -> ViT Backbone -> Reducer -> VQC -> Head -> Output
  (click any node to see formulas + real values for your image)

Grad-CAM Branch:   Fundus Mask -> Backprop -> Heatmap
TCAV Branch:       Concept Scoring -> Per-Concept Maps -> Multi-Color Merge
DME Branch:        Same Image -> 2nd Model -> DME Result
```

---

## 5. Digital Twin Feature

**Phase 2 feature** - an interactive SVG model of the eye that zooms into each anatomical layer and shows AI findings at that level.

```
Eye layers (click any ring to zoom in):

  +--Cornea (outer ring)-----------------------------+
  |  +--Lens--------------------------------------+  |
  |  |  +--Vitreous---------------------------+  |  |
  |  |  |  +--Retina (purple) ---- [*][*][o]  |  |  |
  |  |  |  |  +--Macula (amber) ------- [~]  |  |  |
  |  |  |  |  |  +--Optic Disc (orange)+  |  |  |  |
  |  |  |  |  |  +--------------------+  |  |  |  |
  |  |  |  |  +-----------------------------+  |  |
  |  |  |  +------------------------------------+  |
  |  |  +---------------------------------------+  |
  |  +------------------------------------------+  |
  +--------------------------------------------------+

  [*] = Microaneurysm (red dot)    grade from real TCAV score
  [o] = Hemorrhage (dark red)      grade from real TCAV score
  [~] = Macular oedema (blue)      only if DME detected

  Click Run Scan: layers reveal one by one (cybersecurity scan animation)
  Click finding dot: clinical explanation + severity
  Bottom strip: all 6 layers with severity color at a glance
  Patient explanation: plain-language description per layer

  Demo mode (no upload) : Grade 2 defaults
  Live mode (after upload): real TCAV scores, real grade, real DME result
```

---

## 6. What Every File Does

### Core source

```
config.yaml                    Single config: seed, batch_size, lr, quantum params, paths

models/
  backbone.py                  ViTClassifier: ViT-Base/16 + Linear(768->5)
  hybrid.py                    QuantumRetinaX: loads vit_baseline, freezes it, adds VQC stack
  quantum.py                   QuantumLayer: 4-qubit PennyLane circuit (parameter-shift)
  vqc_only.py                  VQCOnly: tiny CNN encoder + VQC baseline (no ViT)

training/
  train.py                     Phase 2: ViT fine-tuning -> weights/vit_baseline.pth
  train_hybrid.py              Phase 3: VQC stack training -> weights/hybrid_vit_vqc.pth
  train_dme.py                 DME model -> weights/dme_vqc.pth
  train_balanced.py            Class-balanced variant -> weights/hybrid_best.pth
  train_g1boost.py             Grade-1 sensitivity boost -> weights/hybrid_g1boost.pth
  metrics.py                   compute_metrics(): acc/F1/AUC/kappa/G1-sens
  cross_eyepacs.py             EyePACS 5k zero-shot evaluation

data/
  loaders.py                   build_aptos_loaders(), build_dme_loaders()
  transforms.py                Train: RandomFlip+Rotate+ColorJitter. Test: Resize+Normalize

backend/main.py                FastAPI: loads 3 models + 4 TCAV probes, /health + /predict

frontend/src/
  main.tsx                     React Router: / -> Landing, /app -> App
  Landing.tsx                  Product landing page (12 sections, dark purple theme)
  App.tsx                      Clinical dashboard (9 sidebar pages)
  DigitalTwin.tsx              Eye Digital Twin (SVG, real API data)

xai/
  gradcam_demo.py              Grad-CAM grid for all 5 DR grades
  tcav_demo.py                 Train + pickle TCAV probes (run once before backend)
  faithfulness_validation.py   IoU/recall/precision vs IDRiD masks

experiments/scripts/           All eval scripts (see Section 10)
results_revision/              Reviewer-response experiment outputs
paper/final/                   Two IEEE papers (docx)
```

### Weights

| File | Size | Acc | Kappa | Description |
|---|---|---|---|---|
| `vit_baseline.pth` | 329.6 MB | 78.4% | 0.881 | ViT-only, seed 42 |
| `hybrid_vit_vqc.pth` | 329.6 MB | **81.6%** | **0.871** | Main model |
| `dme_vqc.pth` | 329.6 MB | 89.1% | - | DME detection |
| `hybrid_best.pth` | 329.6 MB | 81.5% | 0.872 | Balanced training |
| `hybrid_g1boost.pth` | 329.6 MB | 74.6% | 0.762 | G1-sens 78.6% |
| `vit_improved.pth` | 329.6 MB | 80.0% | 0.870 | 4-layer unfreeze |

> Weights are gitignored (too large). Train them yourself or contact the authors.

---

## 7. Quick Start - Run in 2 Commands

**Prerequisites:** Python 3.12, Node 18+, trained weights in `weights/`

**Step 1 - Backend (new terminal):**
```bash
cd J:\research\QuantumRetina-X
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
# Wait for: INFO: Uvicorn running on http://127.0.0.1:8000
```

**Step 2 - Frontend (new terminal):**
```bash
cd J:\research\QuantumRetina-X\frontend
npm run dev
# Open: http://localhost:5173
```

Or double-click the batch files:
- `run_backend.bat`
- `run_frontend.bat`

---

## 8. Full Setup

**Clone:**
```bash
git clone https://github.com/Alvin20082003/quantum-retina-x.git
cd quantum-retina-x
```

**Python environment:**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

**Python packages:**
```bash
# GPU (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-only (VQC always runs on CPU anyway)
pip install torch torchvision

pip install pennylane pennylane-lightning
pip install fastapi uvicorn
pip install transformers
pip install scikit-learn scipy numpy pandas
pip install pillow opencv-python
pip install pytorch-grad-cam
pip install pyyaml
```

**Frontend packages:**
```bash
cd frontend && npm install && cd ..
```

**Build TCAV probes (once, before first backend start):**
```bash
python xai/tcav_demo.py
# writes experiments/results/tcav/probe_*.pkl
```

---

## 9. Train the Models

**Smoke test first (4 batches, confirms pipeline works):**
```bash
python training/train.py --smoke
python training/train_hybrid.py --smoke
python training/train_dme.py --smoke
```

**Full training:**
```bash
# Phase 2: ViT-only baseline (~70 s/epoch on RTX 3060, 20 epochs)
python training/train.py

# Phase 3: ViT+VQC hybrid (~90 s/epoch - VQC runs on CPU)
python training/train_hybrid.py

# DME model
python training/train_dme.py
```

---

## 10. Run Evaluations

Run all from the **project root**:

```bash
# Main model on APTOS test set
python experiments/scripts/eval_hybrid.py

# Multi-seed reproducibility (~45 min)
python experiments/scripts/eval_multiseed.py

# Zero-shot: IDRiD + Messidor
python experiments/scripts/eval_cross_dataset_full.py

# Zero-shot: EyePACS 5k subset
python experiments/scripts/eval_eyepacs.py

# Reproduce all 10 published numbers
python results_revision/scripts/phase0_reproduce.py

# Grad-CAM visualisations
python xai/gradcam_demo.py
```

---

## 11. Datasets

| Dataset | Images | Use |
|---|---|---|
| APTOS 2019 | 3,662 | Train + test (2563/549/550, seed 42) |
| IDRiD | 455 | Zero-shot cross-dataset, DME labels |
| Messidor-2 | 1,744 | Zero-shot cross-dataset, DME labels |
| EyePACS | 35,126 matched | Zero-shot evaluation (5k subset) |

**Expected layout:**
```
data/
  archive/            APTOS 2019
    colored_images/
      No_DR/  Mild/  Moderate/  Severe/  Proliferate_DR/
    train.csv         id_code, diagnosis (0-4)
  idrid/
    *.jpg
    labels.csv        image_id, grade
  messidor/
    *.png
    labels.csv        image_id, grade
    messidor_data.csv id_code, adjudicated_dme
  eyepacs/
    *.jpeg
    trainLabels.csv   image, level
```

---

## 12. Research Papers

| Paper | Author | Conference Track | File |
|---|---|---|---|
| Hybrid ViT+VQC for Diabetic Retinopathy Grading | Jenna Therese | Track 02 - Advanced Computing | `paper/final/QuantumRetina_Paper1_FINAL.docx` |
| DR+DME Detection with Explainable AI | Alvin Sudhan | Track 01 - Intelligent Systems | `paper/final/QuantumRetina-X_Paper2_IEEE_v2.docx` |

**6 research contributions:**
1. First ViT + VQC architecture for DR and DME early detection
2. Quantum latent space for micro-lesion feature separation
3. Dual XAI (Grad-CAM + TCAV) in a quantum medical model
4. Four-dataset cross-validation across 3 countries
5. Low-data quantum learning efficiency analysis
6. Eye Digital Twin for patient education

---

## Tech Stack

| Layer | Technology |
|---|---|
| Quantum circuit | PennyLane 0.36, default.qubit, parameter-shift |
| Backbone | google/vit-base-patch16-224-in21k (HuggingFace) |
| Deep learning | PyTorch 2.5.1 + CUDA 12.4 |
| Explainability | pytorch-grad-cam + custom TCAV probes |
| Backend | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript + Vite 8 + Tailwind CSS v4 |
| Charts | Recharts |
| Icons | Lucide React |

---

## Citation

```bibtex
@inproceedings{quantumretinax2026,
  title  = {QuantumRetina-X: Hybrid Vision Transformer and VQC for Diabetic Retinopathy Grading},
  author = {Jenna Therese and Alvin Sudhan},
  year   = {2026}
}
```

---

*QuantumRetina-X is a research prototype. Not a medical device. Not for clinical use.*
