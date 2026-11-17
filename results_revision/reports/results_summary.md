# Results Summary - Revision

## Phase 0: Reproduced Results (seed 42, APTOS 550)
| Model | Accuracy | Macro-F1 | AUC | Kappa | G1-Sens |
|---|---|---|---|---|---|
| ViT-only | 0.7836 | 0.6376 | 0.9363 | 0.8811 | 0.7321 |
| ViT+VQC  | 0.8164 | 0.6588 | 0.8938 | 0.8711 | 0.5536 |

## Low-Data (100 samples, seed 42)
| Model | Acc | Kappa | G1-Sens |
|---|---|---|---|
| ViT-only | 0.4309 | 0.5205 | 0.0714 |
| ViT+VQC  | 0.5982 | 0.7614 | 0.0000 |

## A1: 100-Sample Grade Distribution
{'0': 49, '1': 10, '2': 27, '3': 5, '4': 8}
G1 count in 100-sample subset: 10

## A2: 56 True G1 Predicted As
ViT-only-low: G0=34, G1=4, G2=8, G3=9, G4=1
ViT+VQC-low:  G0=18, G1=0, G2=26, G3=0, G4=12

## C1: Trainable Parameters
ViT-only=86,393,093 | VQC-only=5,269 | ViT+VQC=3,125 | Classical-A=3,101 | Classical-B=3,117

## C2: Classical Baselines (mean+-std, 3 seeds)
Classical-A acc: 0.8121+-0.0058 | Classical-B acc: 0.8158+-0.0110
ViT+VQC mean acc: 0.8752+-0.0513

## C3: Runtime (NVIDIA RTX 3060 Laptop GPU)
Head-only ViT: 0.096 ms/img | Head-only ViT+VQC: 16.173 ms/img
Full ViT: 19.290 ms/img | Full ViT+VQC: 38.983 ms/img
VQC simulator: default.qubit (PennyLane), parameter-shift, CPU

## C4: Statistics
McNemar (seed42): b=42, c=24, p=0.0364 (significant at 0.05)
Bootstrap 95% CI Acc ViT-only: [0.7491, 0.8182]
Bootstrap 95% CI Acc ViT+VQC:  [0.7855, 0.8491]
Bootstrap 95% CI Kap ViT-only: [0.8494, 0.9095]
Bootstrap 95% CI Kap ViT+VQC:  [0.8368, 0.9034]
3-seed t-CI ViT-only acc: 0.8061+-0.0211 [0.7537, 0.8584]
3-seed t-CI ViT+VQC  acc: 0.8752+-0.0513 [0.7477, 1.0026]
Paired t-test (acc, 3 seeds): t=3.7937, p=0.0630

## B: EyePACS Version Confirmed = v1
Reproduced: ViT-only 0.6422/0.3520/0.7245/0.4119/0.1810
Reproduced: ViT+VQC  0.6304/0.3249/0.6033/0.4123/0.1897

## B: Domain Adaptation Delta (accuracy vs zero-shot)
IDRiD   ViT-only: ZS=0.4286 AdaBN=+0.0330 CORAL=-0.0857
IDRiD   ViT+VQC:  ZS=0.4352 AdaBN=-0.0110 CORAL=-0.2549
Messidor ViT-only: ZS=0.5940 AdaBN=-0.0103 CORAL=-0.0109
Messidor ViT+VQC:  ZS=0.5831 AdaBN=-0.0419 CORAL=-0.2001
EyePACS ViT-only: ZS=0.6422 AdaBN=-0.0520 CORAL=+0.0920
EyePACS ViT+VQC:  ZS=0.6304 AdaBN=-0.0410 CORAL=-0.5882

## Skipped (time constraint)
B1 colour normalisation (Reinhard/Ben Graham/CLAHE): SKIPPED
B3 fine-tuning with target labels: SKIPPED
A3 class-balancing strategies: SKIPPED
See run_log.md for details.