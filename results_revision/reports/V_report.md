# V_report.md — Verification Report: lowdata_vit_100.pth and lowdata_hybrid_100.pth

Generated: PART 1 investigation

## 1. What training script produced these weights?

	raining/lowdata_experiment.py is the only script in the repo that writes
weights/lowdata_vit_{pct}.pth and weights/lowdata_hybrid_{pct}.pth.
Git log shows this file was committed in the initial commit (3987900).
No second lowdata training script exists.

## 2. What data were they trained on?

lowdata_experiment.py uses **EyePACS**, not APTOS.

Pipeline:
1. Load data/eyepacs/trainLabels.csv, drop unmatched images.
2. Stratified split: 	rain_test_split(df, train_size=5000, stratify=diagnosis, random_state=42)
   → df_sub = 5,000 EyePACS images.
3. Second split: 	rain_test_split(df_sub, test_size=0.3, stratify=diagnosis, random_state=42)
   → train_full ≈ 3,500 images, test_df ≈ 1,500 images.
4. Loop over ractions = [1.0, 0.5, 0.25, 0.1].
   pct = int(frac * 100) → 100, 50, 25, 10.
5. For frac=1.0 (pct=100): 	rain_sub = train_full (no further sub-sampling) → **~3,500 images**.
6. Saves to weights/lowdata_vit_100.pth and weights/lowdata_hybrid_100.pth.

**Conclusion: both files were trained on approximately 3,500 EyePACS images (the 70% split
of a 5,000-sample stratified EyePACS subset), evaluated on a held-out EyePACS test set of
approximately 1,500 images. The name "_100" means 100% of the 3,500-image EyePACS training
pool, NOT 100 APTOS samples.**

No results_log.csv entry was written for these runs (the log contains no lowdata rows),
so no runtime or per-class information beyond the phase0_reproduce.py evaluation is available.

## 3. Backbone source

| Weight file | Backbone source | Evidence |
|---|---|---|
| lowdata_vit_100.pth | **ImageNet in21k** (via ViTClassifier(nc) default) | 198/200 backbone tensors differ from it_baseline.pth by up to 0.0188; the 2 identical tensors are ackbone.pooler.* which are not used in forward(); the differing tensors are encoder dense weights that it_baseline has APTOS-finetuned |
| lowdata_hybrid_100.pth | **weights/vit_baseline.pth** (loaded by QuantumRetinaX.__init__) | 200/200 backbone tensors are bit-for-bit identical to it_baseline.pth (max diff = 0.000000) |

ViTClassifier in models/backbone.py calls ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") and does NOT load it_baseline.pth. lowdata_experiment.py does NOT inject a pre-trained backbone path for ViTClassifier. Therefore lowdata_vit_100.pth started from in21k weights, while lowdata_hybrid_100.pth started from the APTOS-finetuned backbone (because QuantumRetinaX explicitly loads weights/vit_baseline.pth).

Both backbones were frozen during EyePACS training (lowdata_experiment.py sets equires_grad=False for all backbone params; confirmed by backbone tensors being unchanged from their starting point).

## 4. Is the "100 APTOS samples" labelling supported?

**No.** The earlier claim in A_explanation.md that these weights were trained on "100 APTOS samples" is false. These weights were trained on ~3,500 EyePACS images. The phase0_reproduce.py script loads these weights and evaluates them on the APTOS test set (550 images), which is a cross-dataset evaluation, not an APTOS low-data experiment.

The results reported as "Low-data ViT-only (100)" (acc=43.1%, kappa=0.521, G1-sens=7.1%) and "Low-data ViT+VQC (100)" (acc=59.8%, kappa=0.761, G1-sens=0.0%) represent: a model trained on ~3,500 EyePACS images, evaluated zero-shot on the APTOS test set. These are valid cross-dataset transfer results but cannot be presented as a 100-sample APTOS low-data experiment.

## 5. Action

Part 2 below runs the real 100-sample APTOS experiment from scratch with correct labelling.
Results saved to esults_revision/lowdata_real/.
