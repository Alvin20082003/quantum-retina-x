# Run Log

## Environment
- Python: 3.12.3
- PyTorch: 2.5.1+cu124
- PennyLane: 0.36.0
- NumPy: 2.5.1
- scikit-learn: 1.9.0
- Device: NVIDIA GeForce RTX 3060 Laptop GPU
- Date: 2026-10-23 14:19

## Tasks Completed
- Phase 0: PASSED (all 10 checks within 0.3-point tolerance)
- Task A: A1 distribution, A2 confusion matrices + per-class, A4 explanation
- Task C1: Parameter counts
- Task C4: McNemar, bootstrap CI, 3-seed t-CI, paired t-test
- Task C2: Classical baselines (BaseA, BaseB) seeds 42/123/456
- Task B: Zero-shot IDRiD/Messidor/EyePACS confirmed; AdaBN and CORAL computed
- Task C3: Runtime benchmarks

## Deviations
- EyePACS: Used exact original 5k subset via train_test_split(seed=42, stratify, n=5000)
  matching cross_eyepacs.py exactly. This IS the original subset.
- A1 100-sample distribution: approximated proportional sampling with seed=42
  (exact lowdata training script not found; approximation gives 99 samples not 100)

## Skipped Tasks
- B1 Colour normalisation (Reinhard, Ben Graham, CLAHE): SKIPPED — time constraint
- B3 Fine-tuning with target labels: SKIPPED — time constraint (est. 10+ hours)
- A3 Class-balancing strategies (30 VQC runs): SKIPPED — time constraint (est. 18 hours)

## Commands
Phase 0:  python -u results_revision/phase0_reproduce.py
Task A:   python -u results_revision/task_a.py
Task C1:  inline python -c (see scripts)
Task C4:  inline python -c
Task C2:  python -u results_revision/task_c2.py
Task B:   python -u results_revision/task_b_eyepacs.py (5k only after IDRiD/Messidor inline)
Task C3:  inline python -c
