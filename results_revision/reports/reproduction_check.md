# Phase 0 Reproduction Check

## Main Results (seed 42, 550-sample APTOS test)

| Model | Acc (got/expected) | Kappa | AUC | G1-Sens |
|---|---|---|---|---|
| ViT-only | 0.7836 / 0.7840 | 0.8811 / 0.8810 | 0.9363 / 0.9360 | 0.7321 / 0.7320 |
| ViT+VQC  | 0.8164 / 0.8160 | 0.8711 / 0.8710 | 0.8938 / 0.8940 | 0.5536 / 0.5540 |

## Test Grade Support
Got: [271, 56, 150, 29, 44] — Expected: [271, 56, 150, 29, 44] — Match: OK

## Low-Data Results
| Model | Acc (got/expected) | Kappa |
|---|---|---|
| ViT-only (100) | 0.4309 / 0.4310 | 0.5205 / 0.5210 |
| ViT+VQC (100)  | 0.5982 / 0.5980 | 0.7614 / 0.7610 |

## EyePACS Version
Logs reproduce: **v1 (64.2/0.352/0.725)** — confirmed by exact 5k subset reproduction.

## Verdict: PASSED — all values within 0.3-point tolerance.
