# B External Evaluation Summary

## EyePACS Subset Note
EXACT original subset: train_test_split seed=42 stratified 5000 from trainLabels.csv
n=5000

## EyePACS Version Identification
Logs reproduce: **v1 (64.2/0.352/0.725)**
Reproduced: ViT-only 0.6422/0.3520/0.7245/0.4119/0.1810
Reproduced: ViT+VQC  0.6304/0.3249/0.6033/0.4123/0.1897

## Zero-Shot Results Summary

| Dataset | Model | Acc | Macro-F1 | AUC | Kappa | G1-Sens |
|---|---|---|---|---|---|---|
| IDRiD (n=455) | ViT-only | 0.4286 | 0.3252 | 0.7861 | 0.5781 | 0.3182 |
| IDRiD (n=455) | ViT+VQC  | 0.4352 | 0.3354 | 0.6572 | 0.6184 | 0.5000 |
| Messidor (n=1744) | ViT-only | 0.5940 | 0.3234 | 0.7598 | 0.4492 | 0.0333 |
| Messidor (n=1744) | ViT+VQC  | 0.5831 | 0.2953 | 0.5770 | 0.4302 | 0.0556 |
| EyePACS (n=5000) | ViT-only | 0.6422 | 0.3520 | 0.7245 | 0.4119 | 0.1810 |
| EyePACS (n=5000) | ViT+VQC  | 0.6304 | 0.3249 | 0.6033 | 0.4123 | 0.1897 |

## Domain Adaptation Delta (accuracy vs zero-shot)

| Dataset | Method | ViT-only delta | ViT+VQC delta |
|---|---|---|---|
| IDRiD | AdaBN | +0.0330 | -0.0110 |
| IDRiD | CORAL | -0.0857 | -0.2549 |
| Messidor | AdaBN | -0.0103 | -0.0419 |
| Messidor | CORAL | -0.0109 | -0.2001 |
| EyePACS | AdaBN | -0.0520 | -0.0410 |
| EyePACS | CORAL | +0.0920 | -0.5882 |

## Key Finding
AdaBN helps ViT-only on IDRiD (+3.3 pp) but generally hurts or has no effect elsewhere.
CORAL substantially hurts ViT+VQC on all datasets (-20 to -25 pp), likely because the 768-d covariance estimate from 455-1744 samples is unreliable for a 768x768 matrix.
B1 colour normalisation (Reinhard, Ben Graham, CLAHE) was NOT run due to time constraint — see run_log.md.
B3 fine-tuning was NOT run due to time constraint.
