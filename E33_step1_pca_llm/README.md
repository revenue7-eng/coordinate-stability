# Experiment 33: Step 1 PCA Diagnostic — Last-Token Confound and Pole Stability

## What this tests
Whether the leading PCs of an LLM residual stream encode semantic category structure, or whether the apparent structure is an artifact of the last-token effect; and whether the observed category "nodes" at PC poles are stable.

## Setup
- Activation source: `yadro_phase2`, Step 0 (12 Apr 2026)
- Models: Qwen2.5-3B, Gemma2-2B, OLMo-1B, Falcon-1B, Pythia-1.4B
- Data: 80 prompts across 8 categories, residual stream at the last-token position
- Layers: 5 per model (Path 2); final layer (slices 1, 1b, 1c, 1d, 1e, 2)
- PCA: 40 components (25 in slice 2, for speed); exact SVD (`svd_solver=full`)
- Residualization: linear regression on one-hot last token (41 unique tokens over 80 prompts)
- LDA: 5-fold stratified CV
- Bootstrap: 30 resamples, 70% of prompts, seed=42

## Results

### Path 2 — layer by layer, 5 models
| | qwen | gemma | olmo | falcon | pythia |
|---|---|---|---|---|---|
| R2(token) on top-7 PCs, layer 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| R2(token) on top-7 PCs, final | 0.727 | 0.679 | 0.724 | 0.684 | 0.758 |
| delta R2(token) | -0.273 | -0.321 | -0.276 | -0.316 | -0.242 |
| LDA on 15 PCs after residualization, final | 0.450 | 0.400 | 0.438 | 0.475 | 0.425 |
| delta LDA after residualization | +0.238 | +0.175 | +0.226 | +0.250 | +0.237 |

R2(token) falls monotonically with depth in all five models; LDA on the residuals rises monotonically. Step 1 PCA was run on middle layers (Qwen 18/36, Gemma 13/26, OLMo 8/16, Falcon 11/22, Pythia 16/32) — a consistent middle-layer choice, not a Qwen-specific one.

### Slices 1, 1b, 1c, 1d, 1e — pole structure on residuals
- PAIRED PCs (both poles homogeneous, H < 0.3): 1-4 on raw PCA, **zero in all five models on residuals**
- ASYMMETRIC PCs (one pole homogeneous): 2-7 on residuals, varying by model
- Asymmetric poles by category, summed over 5 models: code 8, emotional 7, abstract 4, factual 3, logical 2, spatial 1, narrative 0, ethical 0
- Within-category cosine cohesion on final-layer residuals, averaged over 5 models: emotional 0.128, spatial 0.086, code 0.086, abstract 0.083, factual 0.015, logical 0.003, narrative 0.000, ethical -0.019
- Cohesion vs number of asymmetric appearances: Spearman rho = 0.826, Pearson r = 0.746, N = 8 categories. Permutation null (200 label shuffles at fixed geometry, 163 valid): mean -0.06, sd 0.40, q95 = 0.58, max = 0.76; 0 of 163 draws reached the observed value, p_perm = 0.006 (conservative, (1+k)/(1+n)). Leave-one-model-out gives rho 0.675-0.819, sign unchanged in all five subsets. The parametric p from `spearmanr` is not used at N = 8. Caveat: the code/spatial rank is decided by a gap of 1.27e-4, roughly 120x below the between-model SEM (0.012 for code, 0.018 for spatial). Source: `results/srez1e_output.txt`

### Slice 2 — bootstrap stability of the nodes
- 30 resamples at 70% of prompts, top-15 PCs after PCA (25 components) and residualization
- Stable category (present in >= 70% of bootstraps): code in Falcon only. No other model-category pair qualifies.
- Emotional is not stable in any of the five models.

## Files
- `code/path2_layers.py` — layer-by-layer R2(token) and LDA across 5 models
- `code/pc_polarities.py` — top-5 PCs by 1D LDA on residuals, and what sits at their poles
- `code/srez1_polarity.py` — PC classification: PAIRED / ASYMMETRIC / RANDOM
- `code/srez1b_asym.py` — which category separates asymmetrically
- `code/srez1c_cohesion.py` — within-category cosine cohesion
- `code/srez1d_corr.py` — formal correlation of cohesion with appearances
- `code/srez1e_robustness.py` — permutation null, leave-one-model-out, code/spatial margin
- `code/srez2_bootstrap.py` — node stability under bootstrap
- `code/analiz_kod.py` — original 25 Apr source, all diagnostics in one file
- `results/path2_output.txt` — per-layer tables
- `results/pc_polarities_output.txt` — poles of the top-5 PCs on residuals, 5 models
- `results/srez1_output.txt` — PAIRED / ASYM / RANDOM by model
- `results/srez1b_output.txt` — asymmetric PC detail
- `results/srez1c_output.txt` — cohesion table
- `results/srez1d_output.txt` — Spearman and Pearson correlation
- `results/srez1e_output.txt` — robustness checks for the slice 1d correlation
- `results/srez2_output.txt` — bootstrap stability table
- `results/srez3_pc_extremes_qwen.txt` — poles of all 40 PCs, Qwen (raw Step 1 PCA)
- `results/srez3_pc_extremes_gemma.txt` — poles of all 40 PCs, Gemma (raw Step 1 PCA)

## Additional diagnostics in `analiz_kod.py`
Reproducible from the original file but not split into separate scripts:
- `srez_surface_correlation` — correlation of leading PCs with prompt length, punctuation count, casing. Used as a baseline control: 9 surface features give LDA = 0.375, the bar that semantic axes must clear.
- `srez_dim_vs_lda` — LDA as a function of PC count, 1 to 40. Used to pick the 7-PC and 15-PC cutoffs in the tables above.
- `srez_pc_extremes` — poles of all 40 PCs for Qwen and Gemma, saved to `results/`. Qualitative observation material, not a quantitative result.
- `srez_invariance` — invariance of the leading PCs across models.

## Facts
Ф47, Ф48, Ф49, Ф50, Ф51, Ф52, Ф53, Ф54, Ф55

## Note
This is a diagnostic experiment, not an intervention. It reanalyses existing activations from `yadro_phase2` (Step 0, 12 Apr 2026); no new training was done.

To reproduce: set `YADRO_DATA` to the `yadro_phase2` directory and `YADRO_OUT` to the output directory, then run any script in `code/`. Outputs in `results/` were regenerated on 21 Aug 2026 with numpy 2.5.2 / scipy 1.18.0 / scikit-learn 1.9.0.

## Limitations

**Determinism.** Until 21 Aug 2026 every `PCA` call used the default `svd_solver=auto`, which selects randomized SVD for this data shape and carries no seed. Repeated runs of identical code on identical input produced different numbers: the first mismatch in explained variance appears at component 16-17, and the last components bear little relation to the exact ones (|cos| 0.12-0.79 at component 40). All calls now pass `svd_solver=full`. Figures published before that date are single draws and are not reproducible; the current outputs are exact.

**Resolution ceiling.** With 80 samples and 40 components, components 11-25 lie in a band of 0.0199-0.0097 explained variance, with neighbours differing in the third or fourth decimal. Anything downstream of the tail beyond ~15 components is poorly determined by the data itself, not merely by the solver. This affects `resid7`/`resid15`, the cohesion figures and the asymmetric-loading counts.

**Step 0 provenance.** `step1_pca/step1_pca_results.pkl` was produced outside this repository. Per the note above it was computed on middle layers (Qwen 18/36); the stored subspace is closest to layer 18 of the saved activations (mean principal-angle cosine 0.907) but does not match exactly, and the layer index is not recorded in the pickle. Slices reading `pca_results` — slice 3 and the related `analiz_kod.py` functions — are reproducible from that pickle only, not from the raw activations.

**Partial coverage in `analiz_kod.py`.** `srez_by_layer` is invoked for Qwen only; the calls for the other four models are commented out in the source.

Methodological consequence for the next step: the current dataset (80 prompts with heterogeneous syntactic endings) is exhausted as a test of semantic geometry. Step 2 needs controlled prompts — matched token length, a single shared final token or ending — and activations taken from the final layer rather than a middle one.

## Related experiments
- E27 — drift to quality correlation (methodologically related: reanalysis of existing data)
- Controlled-prompt Step 0, planned, unnumbered — motivated by the conclusions above
