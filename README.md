# QuantumRetina-X

A hybrid quantum-classical AI system for diabetic retinopathy (DR) grading and macular edema (DME) detection.

Combines a frozen **Vision Transformer (ViT-Base/16)** backbone with a **4-qubit Variational Quantum Circuit (VQC)** trained on APTOS 2019. The full model has only **3,125 trainable parameters** — the backbone is 86.4 M frozen weights — and achieves **81.6% accuracy, kappa=0.871** on 5-class DR grading. A clinical React web app lets you upload a fundus image and see the diagnosis, Grad-CAM heatmap, per-lesion localization, and TCAV concept scores live.

---

## Table of contents

1. [What every file is and what it does](#1-what-every-file-is-and-what-it-does)
2. [Architecture](#2-architecture)
3. [Key results](#3-key-results)
4. [Prerequisites](#4-prerequisites)
5. [Setup — one time](#5-setup--one-time)
6. [Run the backend](#6-run-the-backend)
7. [Run the frontend](#7-run-the-frontend)
8. [Train the models](#8-train-the-models)
9. [Run evaluations](#9-run-evaluations)
10. [Dataset layout](#10-dataset-layout)
11. [config.yaml reference](#11-configyaml-reference)

---

## 1. What every file is and what it does

### Root

| File | Purpose |
|---|---|
| `config.yaml` | Single source of truth — seed, image size, batch size, LRs, quantum params, paths. Every script reads this. |

### `models/`

| File | Class | What it does |
|---|---|---|
| `backbone.py` | `ViTClassifier` | Loads `google/vit-base-patch16-224-in21k`, adds `Linear(768->5)` head. Phase-2 baseline and frozen backbone source for the hybrid. |
| `hybrid.py` | `QuantumRetinaX` | Loads `vit_baseline.pth`, freezes the backbone, appends `Linear(768->4)` reducer then 4-qubit VQC then `Linear(4->5)` head. Trainable params: 3,125. Also used for DME with `num_classes=1`. |
| `quantum.py` | `QuantumLayer` | 4-qubit PennyLane `default.qubit` circuit. RY encoding, 2 layers RX/RY/RZ + CNOT ring, Pauli-Z readout. Diff method: `parameter-shift`. Always runs on CPU. |
| `vqc_only.py` | `VQCOnly` | Ablation baseline: tiny CNN encoder (Conv->Conv->Pool->Linear) feeds the same VQC. No ViT. 54.6% vs 81.6% accuracy. |

### `training/`

| File | What it trains | Saves to | Epochs |
|---|---|---|---|
| `train.py` | ViT-only Phase 2. Backbone + head, LRs 3e-5/1e-3, best-kappa checkpoint. | `weights/vit_baseline.pth` | 20 |
| `train_hybrid.py` | ViT+VQC Phase 3. Backbone frozen, only reducer+VQC+head trained. | `weights/hybrid_vit_vqc.pth` | 20 |
| `train_dme.py` | DME binary detection. `QuantumRetinaX(num_classes=1)`, BCE + pos_weight, best-AUC. | `weights/dme_vqc.pth` | 20 |
| `train_balanced.py` | ViT+VQC with class-balanced oversampling. | `weights/hybrid_best.pth` | 20 |
| `train_g1boost.py` | Grade-1 sensitivity boosted by 3x G1 loss weight + gated save. | `weights/hybrid_g1boost.pth` | 20 |
| `train_improved.py` | ViT with 4 backbone layers unfrozen + cosine LR. | `weights/vit_improved.pth` | 30 |
| `train_vqc_only.py` | VQC-only ablation baseline. | `weights/vqc_only.pth` | 20 |
| `metrics.py` | `compute_metrics(y_true, y_prob, num_classes)` returns accuracy, macro-F1, AUC (OvR), quadratic-weighted kappa, per-grade sensitivity, Grade-1 sensitivity. Used by every script. | — | — |
| `cross_dataset.py` | Zero-shot evaluation on IDRiD and Messidor. | `experiments/results/cross_dataset/` | — |
| `cross_eyepacs.py` | Zero-shot EyePACS 5k stratified subset, seed 42. | `cross_dataset_summary.csv` | — |
| `lowdata_experiment.py` | Trains both models at 100/50/25/10% of EyePACS training pool. | `weights/lowdata_*.pth` | 8 |

### `data/`

| File | What it does |
|---|---|
| `loaders.py` | `build_aptos_loaders()` reads `data/archive/train.csv`, indexes all images, makes stratified 70/15/15 split (seed from config), returns three DataLoaders + class-weight tensor. `build_dme_loaders()` pools Messidor + IDRiD DME labels, same pattern. |
| `transforms.py` | Train: Resize(224) + RandomHorizontalFlip + RandomRotation(15deg) + ColorJitter(brightness=0.1, contrast=0.1) + ToTensor + ImageNet Normalize. Test: Resize + ToTensor + Normalize only. |

### `utils/`

| File | What it does |
|---|---|
| `config.py` | `load_config(path)` reads config.yaml with PyYAML. |
| `seed.py` | `set_seed(seed)` sets Python, NumPy, and PyTorch random seeds. |
| `logger.py` | `ExperimentLogger` appends a CSV row to `experiments/results/results_log.csv` after every run. Columns: timestamp, run_name, dataset, model, accuracy, macro_f1, auc_roc, kappa, grade1_sensitivity, notes. |

### `backend/main.py`

FastAPI server. On startup it loads **three models** and **four TCAV probes**.

| Loaded file | Variable | Used for |
|---|---|---|
| `weights/vit_baseline.pth` | `vit_model` | ViT-only inference when `model_type=vit` |
| `weights/hybrid_vit_vqc.pth` | `hybrid_model` | Default inference, Grad-CAM, TCAV embedding |
| `weights/dme_vqc.pth` | `dme_model` | DME probability on every upload |
| `experiments/results/tcav/probe_*.pkl` | `tcav_probes` | Per-concept TCAV scores + spatial heatmaps |

Two endpoints:

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status:ok, device:cuda, models_loaded:true}` |
| POST | `/predict?model_type=hybrid` | JSON: grade, confidence, probabilities, Grad-CAM PNG base64, red-free view, fundus mask, lesion multicolor map, 4 per-concept heatmaps, TCAV scores, DME result, pipeline stage values, recommendation, inference time |

**/predict computes in order:**
1. Resize + normalize with ImageNet stats
2. ViT or hybrid forward pass, 5-class probabilities
3. DME forward pass (separate model), sigmoid probability
4. Grad-CAM on `backbone.encoder.layer[-1].layernorm_before`, percentile-stretched 60-99.5%, masked to Otsu fundus ellipse (18% shrink)
5. Red-free view: green channel + CLAHE (clipLimit=3.0, tile 8x8)
6. TCAV: each probe scores the [CLS] embedding of this exact image
7. Per-concept spatial maps: Grad-CAM through each probe score, winner-take-all merge at threshold 0.3

### `frontend/src/App.tsx`

React 19 + TypeScript single-file app. Eight pages:

| Page | What you see |
|---|---|
| Dashboard | Upload zone, then original / Grad-CAM / lesion map 3-panel, grade bar chart, TCAV bars, DR diagnosis card, DME card, risk level, XAI text summary |
| Pipeline | Live forward-pass visualiser — click any stage (Input / Preprocessing / ViT / Reducer / VQC / Head / Output) to see formulas, real pixel histograms, actual 4-dim VQC output values |
| Explainability | 4 individual per-concept heatmaps (microaneurysms purple, hemorrhages red, exudates green, edema blue) |
| Model Insights | Ablation table, training strategy comparison, cross-dataset table |
| Reports | Scan history list, download Grad-CAM PNG + text report |
| Settings | Switch model (quantum hybrid vs ViT-only), heatmap opacity slider |
| About | Architecture summary, paper references |

Tech stack: React 19, TypeScript, Vite 8, Tailwind CSS v4, Recharts, Lucide icons.
Dev proxy: Vite forwards `/api/*` to the backend.

### `xai/`

| File | What it does |
|---|---|
| `gradcam_demo.py` | 4-row x 5-column Grad-CAM grid (2 images per grade, original + heatmap) saved to `experiments/results/gradcam/gradcam_grid.png` and 5 individual PNGs. |
| `tcav_demo.py` | Trains per-concept linear probes on lesion-patch [CLS] embeddings, pickles to `experiments/results/tcav/probe_*.pkl`. Must run once before the backend starts. |
| `faithfulness_validation.py` | Grad-CAM IoU/recall/precision against IDRiD segmentation masks. Result: mean IoU=0.035, recall=0.126, precision=0.054 (n=81). |

### `experiments/scripts/`

Run from the **project root**.

| Script | Evaluates | Key output |
|---|---|---|
| `eval_hybrid.py` | Main ViT+VQC on APTOS test set | Prints metrics, logs to results_log.csv |
| `eval_vqc_only.py` | VQC-only ablation | Same |
| `eval_g1boost.py` | G1-boosted variant | Same |
| `eval_lowdata.py` | EyePACS-trained lowdata weights on APTOS test set | Comparison table |
| `eval_multiseed.py` | Trains both models at seeds 42/123/456 | `experiments/results/multiseed_results.json` |
| `eval_cross_dataset_full.py` | Zero-shot IDRiD + Messidor | Console + CSV |
| `eval_eyepacs.py` | Zero-shot EyePACS 35k | Console + CSV |

### `experiments/results/`

| File / folder | Contents |
|---|---|
| `results_log.csv` | One row per training run: timestamp, model, dataset, all metrics |
| `multiseed_results.json` | Accuracy/F1/AUC/kappa/G1-sens for ViT and hybrid at seeds 42, 123, 456 |
| `multiseed_seed42_saved.json` | Seed-42 per-epoch checkpoint data |
| `cross_dataset/cross_dataset_summary.csv` | Zero-shot results on IDRiD, Messidor, EyePACS |
| `gradcam/` | Grad-CAM PNGs from xai/gradcam_demo.py |
| `tcav/probe_*.pkl` | Serialised sklearn linear probes (one per lesion concept) |
| `attention_maps/` | Attention rollout images per grade |
| `confusion_matrices/` | Confusion matrix PNGs |
| `ablation/` | Ablation experiment outputs |

### `results_revision/`

Reviewer-response experiments. All reproducible from the scripts.

| Sub-folder | Contents |
|---|---|
| `data/` | `phase0_results.json` ground-truth for 10 reproduction checks. `B_external.csv` zero-shot/AdaBN/CORAL on IDRiD/Messidor/EyePACS. `C_*.csv` params, baselines, runtime, stats. `key_numbers.json` lookup for all published numbers. |
| `reports/` | `run_log.md` what was run / skipped. `V_report.md` verification of lowdata weights. `B_summary.md`, `A_explanation.md`, `reproduction_check.md`. |
| `scripts/` | `phase0_reproduce.py` checks all 10 published numbers. `task_a.py` confusion matrices. `task_c2.py` classical baselines. `fix_c3.py` runtime benchmarks. `task_b_eyepacs.py` colour normalisation. `lowdata_real_run.py` 100-sample APTOS experiment. |
| `A_confusion_matrices/` | 4 PNGs: ViT-only seed42, ViT+VQC seed42, both low-data variants |
| `lowdata_real/` | `seed42.json`, `seed123.json`, subset indices JSON files, `summary.md` |

### `weights/`

| File | Size | Acc | kappa | Description |
|---|---|---|---|---|
| `vit_baseline.pth` | 329.6 MB | 78.4% | 0.881 | ViT-only, seed 42, fine-tuned on APTOS |
| `hybrid_vit_vqc.pth` | 329.6 MB | **81.6%** | **0.871** | **Main model** ViT+VQC seed 42 |
| `dme_vqc.pth` | 329.6 MB | 89.1% | — | DME binary, AUC 0.928, sens 78.5%, spec 91.7% |
| `hybrid_best.pth` | 329.6 MB | 81.5% | 0.872 | G1-boost + gated save |
| `hybrid_g1boost.pth` | 329.6 MB | 74.6% | 0.762 | Grade-1 sensitivity maximised to 78.6% |
| `vit_improved.pth` | 329.6 MB | 80.0% | 0.870 | 4-layer unfreeze + CLAHE, 30 epochs |
| `lowdata_vit_100.pth` | 329.6 MB | — | — | ViT on EyePACS 100% fraction (~3,500 images) |
| `lowdata_hybrid_100.pth` | 329.6 MB | — | — | Hybrid on EyePACS 100% fraction (~3,500 images) |
| `vqc_only.pth` | 0 B | 54.6% | 0.289 | VQC-only ablation |

### `paper/`

| Sub-folder | Contents |
|---|---|
| `final/` | `QuantumRetina_Paper1_FINAL.docx` (DR grading, Track 02 Advanced Computing), `QuantumRetina-X_Paper2_IEEE_v2.docx` (DR+DME+XAI, Track 01 Intelligent Systems), `QuantumRetina-X_Phase1_Review_v2.pptx` |
| `figures/` | 61 PNG/PDF publication figures |
| `scripts/` | 25 generate_*.py / make_*.py scripts that produced the figures |
| `unwanted/` | 16 superseded draft docx files |

### `unwanted/`

29 files moved here during cleanup: root-level log files, one-shot diagnostic scripts. Nothing here is needed to run the system.

---

## 2. Architecture

```
Fundus image  224 x 224 x 3
      |
      v
+---------------------------+
|  ViT-Base/16  FROZEN      |  86.4M params, 196 patches of 16x16
|  fine-tuned on APTOS      |  12 transformer layers, 12-head attn
+------------+--------------+
             |  [CLS] token  768-dim
             v
+---------------------------+
|  Reducer  Linear(768->4)  |  trainable
+------------+--------------+
             |  4-dim compressed vector
             v
+------------------------------------------+
|  4-Qubit VQC  PennyLane default.qubit    |  trainable, runs on CPU
|  RY encoding, 2 layers, CNOT ring        |  diff: parameter-shift
|  RX/RY/RZ gates, Pauli-Z readout         |  24 params
+------------+-----------------------------+
             |  4 expectation values in [-1, 1]
             v
+---------------------------+
|  Head  Linear(4->5)       |  trainable
+------------+--------------+
             |
             v
  G0 / G1 / G2 / G3 / G4   5-class DR grade
```

**Trainable params: 3,125**
- Reducer: 768 x 4 + 4 = 3,076
- VQC weights: 2 layers x 4 qubits x 3 gates = 24
- Head: 4 x 5 + 5 = 25

**DME model:** same QuantumRetinaX with num_classes=1, sigmoid output, trained with BCEWithLogitsLoss + pos_weight on 330 Messidor+IDRiD images.

---

## 3. Key results

### APTOS 2019 test set (n=550, seed 42)

| Model | Acc | Macro-F1 | AUC | kappa | G1-Sens |
|---|---|---|---|---|---|
| VQC-only (no ViT) | 54.6% | 0.237 | 0.646 | 0.289 | 0.0% |
| ViT-only | 78.4% | 0.638 | 0.936 | 0.881 | 73.2% |
| **ViT+VQC** | **81.6%** | **0.659** | **0.894** | **0.871** | **55.4%** |

### Multi-seed (seeds 42 / 123 / 456)

| Model | Seed 42 | Seed 123 | Seed 456 | Mean +- std |
|---|---|---|---|---|
| ViT-only acc | 78.4% | 82.5% | 80.9% | 80.6 +- 1.7% |
| ViT-only kappa | 0.881 | 0.895 | 0.894 | 0.890 +- 0.006 |
| ViT+VQC acc | 81.6% | 91.1% | 89.8% | 87.5 +- 4.2% |
| ViT+VQC kappa | 0.871 | 0.940 | 0.936 | 0.916 +- 0.032 |

### DME (n=330, Messidor + IDRiD)

Accuracy 89.1%, AUC 0.928, sensitivity 78.5%, specificity 91.7%.

### Cross-dataset zero-shot transfer

| Dataset | Model | n | Acc | F1 | AUC | kappa | G1-Sens |
|---|---|---|---|---|---|---|---|
| IDRiD (India) | ViT-only | 455 | 42.9% | 0.325 | 0.786 | 0.578 | 31.8% |
| IDRiD (India) | ViT+VQC | 455 | 43.5% | 0.335 | 0.657 | 0.618 | 50.0% |
| Messidor (France) | ViT-only | 1,744 | 59.4% | 0.323 | 0.760 | 0.449 | 3.3% |
| Messidor (France) | ViT+VQC | 1,744 | 58.3% | 0.295 | 0.577 | 0.430 | 5.6% |
| EyePACS (USA) | ViT-only | 35,126 | 63.6% | 0.349 | 0.719 | 0.407 | 17.0% |
| EyePACS (USA) | ViT+VQC | 35,126 | 62.9% | 0.329 | 0.608 | 0.406 | 18.1% |

---

## 4. Prerequisites

| Requirement | Version used |
|---|---|
| Python | 3.12 |
| PyTorch | 2.5.1+cu124 |
| PennyLane | 0.36.0 |
| Node.js | 18+ |
| npm | 9+ |
| CUDA | 12.4 (optional, CPU works) |
| GPU | RTX 3060 Laptop (ViT inference; VQC always uses CPU) |
| RAM | 8 GB min, 16 GB recommended |
| Disk | ~3 GB for weights + datasets |

---

## 5. Setup — one time

**Step 1 — Clone**
```bash
git clone https://github.com/Alvin20082003/quantum-retina-x.git
cd quantum-retina-x
```

**Step 2 — Python virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**Step 3 — Python packages**
```bash
# CUDA 12.4 build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-only (the VQC always uses CPU anyway)
pip install torch torchvision

pip install pennylane pennylane-lightning
pip install fastapi uvicorn
pip install transformers
pip install scikit-learn scipy numpy pandas
pip install pillow opencv-python
pip install pytorch-grad-cam
pip install pyyaml
```

**Step 4 — Frontend packages**
```bash
cd frontend
npm install
cd ..
```

**Step 5 — Build TCAV probes (once, before first backend start)**
```bash
python xai/tcav_demo.py
# writes experiments/results/tcav/probe_microaneurysm.pkl etc.
```

---

## 6. Run the backend

```bash
# From the project root, .venv active
python backend/main.py
```

Or with uvicorn directly:

```bash
uvicorn backend.main:app --reload
```

**Expected startup output:**
```
[hybrid] loaded ViT backbone from weights/vit_baseline.pth
[hybrid] backbone frozen (86.4M params)
[hybrid] trainable params: 3125 (reducer + VQC + head)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Startup takes 15-30 s: three 329.6 MB weight files, then the VQC compiles the PennyLane circuit on the first call.

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```

---

## 7. Run the frontend

The backend must be running before you open the UI.

```bash
cd frontend
npm run dev
# open http://127.0.0.1:5173 in your browser
```

Vite proxies every /api/* request to the backend at port 8000, so there are no CORS issues.

**Production build:**
```bash
cd frontend
npm run build    # output in frontend/dist/
npm run preview  # serves the build at http://127.0.0.1:4173
```

---

## 8. Train the models

Run from the **project root** with .venv active.

**Smoke test first (4 batches, confirms the pipeline works end to end):**
```bash
python training/train.py --smoke
python training/train_hybrid.py --smoke
python training/train_dme.py --smoke
```

**Full training sequence:**
```bash
# Phase 2 — ViT-only baseline, ~70 s/epoch on RTX 3060, 20 epochs
python training/train.py

# Phase 3 — ViT+VQC hybrid, backbone frozen, ~90 s/epoch (VQC on CPU)
python training/train_hybrid.py

# DME binary detection model
python training/train_dme.py

# Ablation variants (optional)
python training/train_vqc_only.py
python training/train_g1boost.py
python training/train_balanced.py
```

Each script saves the best checkpoint (kappa for DR, AUC for DME) to weights/.

---

## 9. Run evaluations

Run from the **project root**.

```bash
# Main hybrid model on APTOS test set
python experiments/scripts/eval_hybrid.py

# Multi-seed reproducibility, ~45 min
python experiments/scripts/eval_multiseed.py

# Zero-shot: IDRiD + Messidor
python experiments/scripts/eval_cross_dataset_full.py

# Zero-shot: EyePACS full 35k
python experiments/scripts/eval_eyepacs.py

# Reproduce all 10 published numbers
python results_revision/scripts/phase0_reproduce.py

# Grad-CAM grid visualisation
python xai/gradcam_demo.py
```

Results land in experiments/results/ and results_revision/data/.

---

## 10. Dataset layout

| Dataset | Images | Used for |
|---|---|---|
| APTOS 2019 | 3,662 | Training + test (2563 / 549 / 550, seed 42) |
| IDRiD | 455 | Zero-shot cross-dataset, DME labels |
| Messidor-2 | 1,744 | Zero-shot cross-dataset, DME labels |
| EyePACS | 35,126 matched | Zero-shot evaluation |

```
data/
  archive/                   APTOS 2019
    colored_images/
      No_DR/  Mild/  Moderate/  Severe/  Proliferate_DR/
    train.csv              id_code, diagnosis (0-4)
  idrid/
    *.jpg
    labels.csv             image_id, grade
    idrid_labels.csv       id_code, Risk of macular edema
  messidor/
    *.png
    labels.csv             image_id, grade
    messidor_data.csv      id_code, adjudicated_dme, adjudicated_gradable
  eyepacs/
    *.jpeg
    trainLabels.csv        image, level
```

---

## 11. config.yaml reference

```yaml
seed: 42                     # used everywhere: splits, weight init, training loops

data:
  image_size: 224
  num_classes: 5             # DR grades 0-4
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std:  [0.229, 0.224, 0.225]

train:
  batch_size: 8
  epochs: 20
  lr_backbone: 3.0e-5        # only used in train.py (ViT fine-tuning)
  lr_head: 1.0e-3            # used in train_hybrid.py and all frozen-backbone scripts
  weight_decay: 1.0e-4
  drop_last: true
  class_imbalance: weighted_loss

quantum:
  n_qubits: 4
  n_layers: 2
  sim_device: lightning.qubit  # config key; quantum.py uses default.qubit in code
  init_std: 0.01

paths:
  results_dir: experiments/results
  weights_dir: weights
```
