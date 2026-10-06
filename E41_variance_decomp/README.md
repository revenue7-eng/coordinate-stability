# E41: Variance decomposition, encoder init x head init (Push-T, gym-pusht)

## What this tests
How much of the spread in best_vp between frozen initialisations belongs to the encoder initialisation and how much to the predictor-head initialisation, in a crossed design of 8 encoder seeds by 5 head seeds at a fixed data seed.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф65 — E41: the spread between frozen initialisations belongs to the encoder
- [verified: python E41_variance_decomp/code/analyze_e41.py over E41_variance_decomp/results/grid.json, 2026-09-14] Crossed grid, 8 encoder seeds x 5 head seeds, data seed 42 fixed, encoder frozen at step 0, one run per cell. Two-way crossed random effects, df 7 / 4 / 28, interaction confounded with the residual. Variance share of best_vp: encoder 0.993, head 0.002, residual 0.005; F_enc = 1021.2, F_head = 4.92.
- [verified: same run] Stable across metric and scale: final_vp 0.994, mean_last3 0.992; on the log scale 0.982, 0.985, 0.978. Within a fixed encoder the five heads spread 1.05x to 1.25x, while encoder levels run 0.002365 to 0.009158, a spread of 3.87x over seeds 1..8 (the raw E40 spread over the same eight is 4.72x).
- [verified: e40_lib.py:120-125 and 136-138] The confound is real in the code: E40 reseeded one stream from init_seed immediately before M(enc, AE(), PR(), SIGReg()), so a single argument drew encoder and head together, and with the encoder frozen before the first step only the head ever trained. It is negligible in magnitude.
- [verified: run_e41.py acceptance cell, bit_exact True, rel diff 0.000e+00] Cell (enc_seed=1, head_seed=None) reproduces E40 init 1 at best_vp = 0.0031448905217346915 exactly, so the seed split did not move the stream and encoder seeds 1..8 are bit-identical to E40 inits 1..8.
- VERDICT: the riskiest assumption carried into Г27 is closed negatively. The spread is not a property of the (encoder, predictor) pair; it is a property of the encoder.

## Ф66 — the encoder level is recoverable from a single run
- [verified: analyze_e41.py, 2026-09-14] corr(E40 best_vp over inits 1..8, E41 encoder level averaged over five heads) = +0.9951 pearson, +0.9762 spearman.
- CONSEQUENCE: a crossed design is not needed for the campaigns that follow. One head per initialisation measures the encoder level to within the head noise, which halves the cost of the Г27 sweep.

## Ф67 — eff_rank is not a confirmed lead for Г27
- [verified: recomputation over E40_init_sweep/results/sweep.json, 2026-09-14] corr(best_vp, eff_rank) = -0.2871 over all ten initialisations, -0.6994 over the eight used in the E41 grid, -0.7005 without init 9, +0.3919 without init 7. Spearman over ten: -0.1030, exact two-sided permutation p = 0.785.
- [verified: analyze_e41.py] Against the head-averaged encoder level over the same eight: pearson -0.7111, spearman -0.4524, exact p = 0.2675. Averaging the head noise out is not what moves the number; the choice of points is.
- [verified: same files] init 9 (eff_rank 1.5380, best_vp 0.001221) and init 7 (eff_rank 1.5755, best_vp 0.008927) share the lowest ranks with opposite outcomes, which is why the correlation swings with the subset.
- [verified: same files] corr(best_vp, R2_readout) = +0.0596 over ten and +0.2307 over the eight; spearman over ten -0.0061, exact p = 1.000. The negative part of Ф64 is unaffected.
- [verified: recomputation over sweep.json, 2026-09-14] With both low-rank extremes removed, the correlation is absent: pearson -0.1843, spearman -0.3095 over the remaining eight; -0.1830 / -0.2500 over seven (also dropping init 10). At n=8 the resolution limit at alpha=0.05, power 0.80 is |rho| = 0.85 (0.89 at n=7), so this is no signal rather than a weak one. The -0.6994 on the grid subset 1..8 is produced by including init 7 and excluding init 9, not by a relation that holds in the bulk.
- VERDICT: eff_rank is not a lead. Seeds 1..8 were chosen for the E41 grid on no criterion related to eff_rank, so reading the stronger correlation on that subset as evidence would be a selection artifact.

## Ф68 — the shape of the spread: two tiers, and the tails carry the structure
- [verified: jackknife over E40_init_sweep/results/sweep.json, 2026-09-14] Over the ten initialisations best_vp spans 7.3086x (CV 0.556, median 0.00352787). Leave-one-out leaves the ratio at 7.3086 for every point except init 7 (3.9544) and init 9 (4.7191). There is no third sensitive point. Removing both extremes: ratio 2.5533, CV 0.256.
- [verified: jackknife over E41_variance_decomp/results/grid.json, 2026-09-14] The same on the eight head-averaged encoder levels: 3.873x (CV 0.519) on best_vp, unchanged under every leave-one-out except e7 (2.001, CV 0.243). On mean_last3: 3.576x (CV 0.505), and 1.858x (CV 0.231) without e7.
- [verified: same file] Selection along the trajectory is not the source of the spread: best_vp equals final_vp in 27 of 40 cells, argmin_ep is 14 or 15 in 37 of 40 (epochs=15), and the ratio loses 7.7% moving to the non-selective mean_last3 (3.873 -> 3.576).
- [verified: Ф65 within-encoder figures] The upper extreme is reproducible rather than a noise point: all five heads on e7 give 0.00896 to 0.00951.
- CONNECTION to Ф67: init 7 and init 9 set the magnitude of the spread here and the direction of the eff_rank correlation there. One property of the data seen twice, not two findings.
- CONSEQUENCE: the headline 7.31x is a distance between tails, not the width of the typical spread. The bulk spreads about 2x at CV ~0.24, robust to dropping any single point; rare initialisations depart by a further factor of ~3. Both tiers must be stated. The tail tier rests on two observations out of ten, which is what E42 must size for.
- CORRECTED BY Ф73 (2026-09-28): the two tails are the two persistence extremes among the ten, init 9 lowest (0.001759) and init 7 highest (0.017602). The two-tier reading describes the distribution of persistence, not a separate structure of best_vp. The CONNECTION to Ф67 becomes the observation in Ф74 that eff_rank is lowest at both persistence extremes.

## Setup
From EXPERIMENTS.md, section E41.

- **Status:** COMPLETE 2026-09-14. Facts: Ф65, Ф66, Ф67. Registry: EVIDENCE.md.
- **Question:** does the E40 spread belong to the encoder, or to the
  (encoder, head) pair? E40 drew both from one seed argument, so the two were
  confounded.
- **Design:** crossed grid, 8 encoder seeds x 5 head seeds, 40 cells, one run per
  cell, data seed 42 fixed, encoder frozen at step 0. Two-way crossed random
  effects, df 7 / 4 / 28, interaction confounded with the residual.
- **Acceptance:** encoder seeds 1..8 reproduce E40 initialisations 1..8
  bit-exactly, so the seed split did not move the stream.
- **Code:** `E41_variance_decomp/code/e41_lib.py`, `run_e41.py`,
  `analyze_e41.py`. **Results:** `E41_variance_decomp/results/grid.json`.

## Files
- `code/__pycache__`
- `code/analyze_e41.py`
- `code/e41_lib.py`
- `code/patch_e41_lib.py`
- `code/patch_registries_e41.py`
- `code/run_e41.py`
- `results/grid.json`
- `results/run_e41.log`

## Facts
Ф65, Ф66, Ф67, Ф68

## How to reproduce
```
python run_e41.py probe     acceptance cell only
python run_e41.py           full grid (resumable via results/grid.json)
python3 E41_variance_decomp/code/analyze_e41.py
```

## Status
Summary table: within an own-latent metric.

- Ф65: no correction recorded in the registry; stands as recorded
- Ф66: no correction recorded in the registry; stands as recorded
- Ф67: no correction recorded in the registry; stands as recorded
- Ф68: no correction recorded in the registry; stands as recorded

<!-- marker: repo-state v3 applied -->
