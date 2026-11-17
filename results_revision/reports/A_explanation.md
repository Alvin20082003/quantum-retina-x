# A4 - Factual Explanation of Low-Data Grade-1 Sensitivity

The 100-sample APTOS training subset (stratified, random_state=42) contains 50 Grade-0, 10 Grade-1, 27 Grade-2, 5 Grade-3, and 8 Grade-4 images, summing to exactly 100.

On the 550-sample APTOS test set, ViT-only trained on this subset correctly classifies 4 of the 56 true Grade-1 images (sensitivity 7.1%), with 34 of the 56 misclassified as Grade-0; ViT+VQC correctly classifies 0 of 56 Grade-1 images (0% sensitivity) and predicts Grade-1 zero times across the entire test set.

ViT+VQC routes 100 of 150 true Grade-2 images to Grade-4, while correctly identifying 268 of 271 Grade-0 images (98.9% sensitivity).

The training subset contains only 10 Grade-1 image(s), compared with 50 Grade-0 images; with this imbalance, both models learn no reliable Grade-1 decision boundary from 10 example(s), whereas the full-data ViT-only model (2,563 training images) achieves 73.2% Grade-1 sensitivity on the same test set.
