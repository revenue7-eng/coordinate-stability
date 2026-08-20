# Pre-E30: Coordinate Drift Pilot on DINOv2

**Status:** COMPLETED (pilot)
**Date:** 25 Apr 2026; sanity baselines corrected 20 Aug 2026
**Type:** Pilot study, not a standalone result

## Goal
Test the coordinate-drift methodology on production-scale self-supervised models before committing compute to a full E30. Specifically: does the latent coordinate system survive between models of the same family (DINOv2), trained by Meta on identical data with an identical algorithm, differing only in capacity?

## Hypothesis
If free latent spaces lose coordinate identifiability — the central claim of the prescribed-axes programme — then DINOv2-small (22M, 384D) and DINOv2-base (86M, 768D) should yield non-alignable coordinate systems on the same images despite identical training data and procedure.

## Setup
- Models: facebook/dinov2-small (22M, 384D), facebook/dinov2-base (86M, 768D)
- Dataset: CIFAR-100 test split, random subset N=500, seed=42
- Embeddings: CLS token from last_hidden_state
- Dimension equalization: PCA (base 768D to 384D)
- Hardware: Windows 10, CPU only, no CUDA
- Inference: 97s (small) + 391s (base)

## Metrics
- Procrustes R2 (raw) — E27 formula, scale-sensitive
- Procrustes R2 (Frobenius-normalized) — scale-invariant
- Linear CKA — rotation-invariant geometry similarity (Kornblith et al. 2019)

## Results
| Metric | Value | Shuffled-pairs floor | Above floor |
|---|---|---|---|
| Procrustes R2 (raw) | 0.645 | 0.203 | 0.555 |
| Procrustes R2 (Frob) | 0.650 | 0.211 | 0.556 |
| Linear CKA | 0.764 | 0.256 | 0.683 |

Explained variance: small 384/384 = 100%; base top 384/768 = 98.3%. All Procrustes and CKA figures are computed on the truncated 384-dimensional space.

## Sanity baselines
The original control compared the embeddings against `np.random.randn`, giving CKA = 0.305. That is not a valid null — Gaussian noise shares neither the spectrum nor the scale of real embeddings. The correct control permutes the rows of one side, preserving everything except the image-to-image correspondence.

Floors under row permutation: R2 raw 0.203, R2 Frob 0.211, CKA 0.256 (20 draws, sd < 0.003).

The positive floor is specific to this geometry. With n=500 points in d=384 dimensions the ratio n/d is 1.3, and an orthogonal d x d matrix has enough freedom to partially fit a random correspondence. On low-dimensional latents with many points — the Push-T regime, d = 3 to 16 — the same estimator has a floor near -0.9. Procrustes R2 has no fixed null; it must be established per geometry.

## Findings
1. Direction matches the hypothesis. CKA exceeds Procrustes R2 by 0.119 raw and 0.128 after floor normalization. Geometry is preserved better than coordinate correspondence — the predicted pattern. The correction did not remove it.
2. Effect is smaller than at small scale. E27 reported R2 = -16.94 on transfer; here R2 = +0.65. Production-scale pretraining (LVD-142M) substantially stabilizes coordinates without making them identical. Note that -16.94 also sits far below the random floor for its geometry and deserves a separate look.

## Limitations
1. Capacity confound: small and base are different models of different capacity, not different seeds. Drift here may mean "base found more structure" rather than "coordinates are unstable".
2. N=500 gives low statistical power. A full experiment needs N >= 5000.
3. CIFAR-100 32x32 upscaled to 224x224 is out of distribution for DINOv2, which was trained on natural images.
4. Single sampling seed, no error bars. A full experiment requires bootstrap.

## What this pilot supports
Sufficient to show the methodology works on production models, that the direction of the effect matches theory, and to justify designing a full E30. Not sufficient to publish as evidence for the drift hypothesis or to make strong claims about the nature of free latent spaces.

## Files
- `code/e30_run.py` — main script
- `results/embeddings.npz` — embeddings (small, base), labels, indices
- `results/results.json` — metrics, config, and corrected sanity baselines

## Related experiments
- E06, E18 — original Procrustes drift on Push-T
- E22 — SIGReg confounds absent
- E27 — drift to val_loss correlation
- E36 — production-scale drift test (PLANNED; this pilot motivates its design)
