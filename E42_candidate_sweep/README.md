# E42: Candidate-predictor sweep for Г27 (Push-T, gym-pusht)

## What this tests
Whether any of four pre-registered cheap properties of an untrained encoder (eff_rank, R2_readout, condition number of the representation covariance, smoothness of the latent dynamics) predicts best_vp across 30 frozen initialisations, with a Bonferroni correction.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф69: E42 pre-registered outcome, three candidates refused, candidate 4 unresolved as registered
- [verified: python3 E42_candidate_sweep/code/analyze_e42.py, 2026-09-28; git status clean after the run] Seed 42, 30 cells, one head each. Continuity with E40 over inits 1..10: 10/10 bit-exact. best_vp spans 7.3086x (CV 0.468, median 0.00294939).
- [verified: same] Candidates against best_vp, as Pearson / Spearman / p_fisher / p_perm: eff_rank -0.2077 / -0.1008 / 0.2734 / 0.59411; r2_readout +0.0235 / +0.0056 / 0.9027 / 0.97725; cond_number -0.0293 / +0.0986 / 0.8788 / 0.60240; smoothness +0.3893 / +0.5648 / 0.0327 / 0.00127. Control rms_norm +0.2210 / +0.0834 / 0.2430 / 0.66115. The decision column at 0.0125 reads no for all four candidates; it applies p_fisher, which the script computes on the Pearson coefficient.
- [verified: same] Jackknife ranges: eff_rank [-0.3029, -0.0275], r2_readout [-0.0548, +0.1374], cond_number [-0.0952, +0.1022], smoothness [+0.2993, +0.5978], rms_norm [-0.0732, +0.3020]. eff_rank, smoothness and rms_norm are each most sensitive to init 7.
- [verified: same] Second data seed 123: 10 cells, spread 7.3135x, CV 0.496. No correlation is computed there by design; the second seed tests reproduction of the spread.
- [verified: EXPERIMENTS.md:594-595; Fisher z of +0.5648 at n=30 gives 0.00088, of +0.3893 gives 0.03272] Candidate 4 fails on Pearson (p 0.0327) and passes on Spearman (Fisher p 0.00088, permutation p 0.00127), recorded unresolved in df7dd76. analyze_e42.py prints the Spearman coefficient and the permutation p but not the Spearman Fisher p.
- STATUS of candidate 4: unresolved as registered. Ф72 removes the need to resolve it: candidate 4 is a function of the control variables.

## Ф70: the validation split behind every representation metric is the split the head was scored on
- [verified: grep -n "NEP" E39_subepoch_freeze_micro/code/analyze_representation.py, line 47: EP, NEP = 15, 200; run_e42.py sets EP, NEP = 15, 200] val_states and the training cells collect the same number of episodes.
- [verified: diff E39_subepoch_freeze_micro/code/e39_lib.py E41_variance_decomp/code/e41_lib.py] The two libraries differ only inside run_subepoch (signature and the enc_seed/head_seed block). collect_gym_data and DS, which val_states imports from e39_lib, are identical to the ones run_subepoch uses.
- [verified: sha256sum E42_candidate_sweep/code/e42_lib.py E41_variance_decomp/code/e41_lib.py] Identical, c41d9434a5fc0ffccec3f29527240547939195651c5b04c078c0f60a4a39f6b9.
- [verified: sweep.json] n_batches is 160 in all 40 E42 cells.
- [verified: persistence_e42.py acceptance] The stored final_vp of s42_i1 and s123_i1 is reproduced from model_final.pt on the rebuilt split with relative difference 0.
- [verified: e41_lib.py run_subepoch] random_split draws from its own torch.Generator seeded with the data seed, before any reseed by enc_seed, and the global stream is restored to the data seed after the model is built. At one data seed every cell shares the validation split and the training batch order; only the encoder initialisation varies.
- CONSEQUENCE: Ф64, Ф67 and Ф69 compute representation metrics on the validation windows the head was scored on.

## Ф71: SIGReg contributes no gradient in the frozen regime
- [verified: python3 -c "... from e42_lib import SIGReg; sum(p.numel() for p in SIGReg().parameters())" -> 0] SIGReg has no parameters.
- [verified: e41_lib.py M.forward and run_subepoch] sl = sig(emb.transpose(0, 1)) depends only on the encoder output. With freeze_frac = 0 the encoder is frozen before the first optimiser step. The action encoder and the predictor do not enter sl.
- CONSEQUENCE: in every cell with freeze_frac = 0 (all of E40, E41, E42) the lam * sl term moves no trained parameter, and the head is trained on the prediction loss alone. Describing this line as a SIGReg-regularised objective is wrong in substance.

## Ф72: candidate 4 is a function of the control variables
- [verified: E42_candidate_sweep/code/e42_candidates.py, def smoothness] smoothness = mean(||z_{t+1} - z_t||) / rms_norm(z3). The pre-registered control rms_norm is its denominator.
- [verified: E42_candidate_sweep/code/persistence_e42.py] persistence is the mean over validation items of mse(emb[:, 2], emb[:, 3]), that is mean ||dz||^2 / 3 at one within-window step.
- [verified: inline python over results/sweep.json and results/persistence.json, 2026-09-28] log(smoothness) against 0.5 * log(3 * persistence) - log(rms_norm): corr +0.997401 at seed 42 (n=30), +0.995406 at seed 123 (n=10). The offset is nearly constant: mean -0.2851 (sd 0.0311) and -0.2710 (sd 0.0302). smoothness is about 0.75 * sqrt(3 * persistence) / rms_norm.
- [verified: E42_candidate_sweep/code/twopredictor_e42.py] Against the residual of log(vp) on log(persistence) and log(rms_norm) together, smoothness gives -0.0276, -0.0209, -0.0247 at seed 42 for best_vp, final_vp, mean_last3.
- [verified: docstring of smoothness in e42_candidates.py] Windows are built at stride 1 and interior states enter the mean up to T times, so the observations behind this candidate are not independent.
- CONSEQUENCE: to within about 3% in the log, candidate 4 is a deterministic function of the pre-registered control and of persistence. Its raw association with best_vp in Ф69 is mechanical, and the Pearson/Spearman question recorded in df7dd76 has nothing left to decide.

## Ф73: persistence orders the frozen initialisations (exploratory, reproduced on a second data seed)
- [verified: e41_lib.py M.forward] The prediction target is tgt = emb[:, 3], the detached output of the same frozen encoder. The encoder sets the scale of the loss it is scored by.
- [verified: persistence_e42.py] corr(log best_vp, log persistence) +0.9708 at seed 42 (n=30) and +0.9789 at seed 123 (n=10). persistence spans 10.0067x and 9.3967x. best_vp / persistence < 1 in all 40 cells.
- [verified: twopredictor_e42.py] OLS slope of log(best_vp) on log(persistence): 0.7790 (R2 0.9425) at seed 42, 0.8100 (R2 0.9583) at seed 123.
- [verified: twopredictor_e42.py] persistence is not a scale quantity: slope of log(persistence) on log(rms_norm) -0.0136, corr -0.0090 at seed 42, where a pure scale quantity would give 2. At seed 123, n=10: slope +1.1617, corr +0.4202.
- [verified: inline python over results/persistence.json and sweep.json, s42_i1..i10 bit-exact with E40] Among E40's ten, init 9 has the lowest persistence and init 7 the highest, and best_vp is monotone in persistence except for one adjacent swap (init 2 and init 3).
- STATUS: not pre-registered. Found at seed 42 and reproduced at seed 123.
- CONSEQUENCE: under this objective best_vp measures mainly how far apart the frozen encoder places consecutive states, which fixes how hard the head's task is. persistence is computed from the frozen encoder before any training, so the statement that nothing cheap orders the initialisations does not hold for this metric.

## Ф74: normalising vp by persistence, and what survives the controls (exploratory)
- [verified: persistence_e42.py] Division over-corrects. OLS slopes of log(vp) on log(persistence) run from 0.7441 to 0.8810 across the three numerators and two seeds, never 1, and corr(log(vp / persistence), log persistence) runs from -0.6627 to -0.8325 in all six combinations.
- [verified: inline python over sweep.json, argmin_ep and final_vp] best_vp is a minimum over 15 epochs: argmin_ep is below 15 in 13 of 40 cells and best_vp / final_vp falls to 0.6643. final_vp and mean_last3 carry no such selection. The spread survives without it: final_vp spans 6.7097x and 7.3135x.
- [verified: persistence_e42.py] On the one-predictor residual the control rms_norm has partial correlation +0.5779 (p 0.0008) with log(best_vp) given log(persistence) at seed 42, stronger than any candidate.
- [verified: twopredictor_e42.py] Adding log(rms_norm) raises R2 by 0.0179 and leaves residual variance 0.0396 at seed 42. Against that residual eff_rank gives -0.4088 (p 0.0299), -0.3503 and -0.3666 for best_vp, final_vp, mean_last3; r2_readout, cond_number and smoothness stay below 0.27 in absolute value.
- [verified: analyze_representation.py eff_rank, p = w / w.sum()] eff_rank is invariant to a global rescaling by construction; at seed 42 it correlates -0.0205 with log persistence and +0.1682 with log rms_norm. It correlates -0.7810 with cond_number, so the two are not independent candidates.
- [verified: inline python over results/persistence.json and sweep.json] Among E40's ten, the two lowest eff_rank values, init 9 (1.5380) and init 7 (1.5755), sit at the two persistence extremes.
- STATUS: the eff_rank figure is the largest of roughly sixty correlations computed across five normalisations, three numerators and four candidates, none named in advance. It is a basis for one pre-registration, not a result. The low-rank-at-both-extremes observation rests on two points of ten, the configuration Ф67 warns about.

## Setup
From EXPERIMENTS.md, section E42.

- **Status:** PLANNED. Tests Г27 as refined 2026-09-14: the target is not a
  property that orders frozen bases in general, but one that explains the tails
  (Ф68).
- **Design:** 30 initialisations at data seed 42, one head each (justified by
  Ф66: the encoder level is recoverable from a single run, so no crossed design
  is needed). Encoder frozen at step 0. Plus 8-10 initialisations at a second
  data seed, run in the same campaign. The second data seed is 123, fixed
  2026-09-15 before any cell of the seed-42 half was read. It is one of the
  five E39 seeds, so E39 representation metrics exist on it for cross-check.
- **What the second data seed does and does not test:** it checks whether the
  spread and the variance shares reproduce off seed 42. It does NOT test any
  correlation: 8-10 points cannot resolve one.

**Candidate properties, closed list, all computed on the untrained encoder before
the first optimiser step, on the full validation split as returned by
`val_states(data_seed)` from E39's `analyze_representation.py`, which is the
same set of states behind Ф64 and Ф67. Not a batch: `eff_rank` and
`R2_readout` are imported from that module rather than reimplemented, so the
new values are the same quantities as 1.5380 and 1.5755. Candidate 4 needs the
window axis that `val_states` flattens away, so it reads the pre-reshape
tensor (Nv, H+2, 5); candidates 1 to 3 read the flattened one:**
1. `eff_rank` of the representation. Included for continuity with Ф67, where it
   was not a lead at n=8-10.
2. `R2_readout`, linear extractability of the true state. Included for continuity
   with Ф64, where corr = +0.060 at n=10.
3. Condition number of the representation covariance, defined as the ratio of
   largest to smallest eigenvalue of the same `np.cov(rep.T)` that `eff_rank`
   reads, so the two candidates are computed on one matrix. Distinct hypothesis from
   eff_rank: sensitive to the worst-conditioned direction rather than to how
   evenly variance is spread. Ill-conditioning is a plausible tail mechanism.
4. Smoothness of the latent dynamics:
   `mean(||z_{t+1} - z_t||) / sqrt(mean(||z_t||^2))` over consecutive states.
   The only candidate motivated by the task rather than by representation theory:
   the head predicts dynamics, and a representation in which the dynamics tear
   should be harder to predict in. Expected sign: positive with `best_vp`.
   Reachability checked before pre-registration: `DS` windows hold H+2 = 5
   consecutive states per item and `__getitem__` returns the window whole,
   so the pairs are within-sample and independent of the loader shuffle
   [e41_lib.py:57-66]. Windows are built at stride 1 and therefore overlap,
   so interior states enter the mean up to five times: the observations
   behind this candidate are not independent, and the jackknife report for
   it says so.

**Control, outside the multiplicity correction:** norm of the encoder output. It
scales the loss directly, so a correlation there would be about units rather than
about basis quality. Reported as a sanity check on whether SIGReg equalises scale
across initialisations; never counted as a hit.

**Statistics, fixed before the run:**
- Pearson and Spearman of each candidate against `best_vp` over all 30 points.
  AMBIGUITY FOUND AFTER THE RUN (2026-09-17): this line names two statistics
  and the next line one threshold, without saying which governs a hit. It
  decided nothing for candidates 1 to 3 and everything for candidate 4, which
  fails on Pearson (p = 0.0327) and passes on Spearman (p = 0.00088 Fisher,
  0.00127 permutation). Recorded here unresolved: choosing either one now
  would be choosing the test after seeing the result. A successor experiment
  names one statistic before it runs.
- Bonferroni over the four candidates: significance threshold alpha = 0.0125.
  At n=30, power 0.80, that resolves |rho| from 0.567 two-sided (uncorrected 0.492).
- Leave-one-out jackknife over the 30 points is reported for every candidate,
  significant or not. It is part of the report, not a response to an inconvenient
  result. Rationale: in this line the correlation has already been shown to be
  set by two points out of ten (Ф67, Ф68).
- The composition of points is not changed after seeing the results. Any subset
  analysis is reported alongside the full-sample figure, never in place of it.

**Declared outcomes:**
- If no candidate clears the corrected threshold: the property that orders frozen
  bases is not a cheap one. That is the registered result of this experiment, not
  a failure of it, and it closes Г27 negatively.
- If a candidate clears it and survives the jackknife: it becomes a lead and Г27
  moves to a confirmatory test on an independent data seed.
- If a candidate clears it and does not survive the jackknife: recorded as
  tail-driven, not as a lead, following the Ф67 precedent.
- Independently of the above, the 30 points measure the distribution of encoder
  levels, which is what the two-tier claim (Ф68) currently rests on with two
  observations out of ten.

**Cost:** about 86 s per cell as measured in E41, so roughly 45 min for the 30
plus about 15 min for the second-seed runs.

## Files
- `code/__pycache__`
- `code/analyze_e42.py`
- `code/e42_candidates.py`
- `code/e42_lib.py`
- `code/persistence_e42.py`
- `code/run_e42.py`
- `code/twopredictor_e42.py`
- `results/persistence.json`
- `results/run_e42.log`
- `results/sweep.json`

## Facts
Ф69, Ф70, Ф71, Ф72, Ф73, Ф74

## How to reproduce
```
python run_e42.py probe      acceptance cell only
python run_e42.py            seed 42 half, resumable via results/sweep.json
python run_e42.py second     second data seed half, once SECOND_SEED is set
python3 E42_candidate_sweep/code/analyze_e42.py
```

## Status
Summary table: persistence orders frozen inits (Ф73).

- Ф69: STATUS of candidate 4: unresolved as registered. Ф72 removes the need to resolve it: candidate 4 is a function of the control variables.
- Ф70: no correction recorded in the registry; stands as recorded
- Ф71: no correction recorded in the registry; stands as recorded
- Ф72: no correction recorded in the registry; stands as recorded
- Ф73: STATUS: not pre-registered. Found at seed 42 and reproduced at seed 123.
- Ф74: STATUS: the eff_rank figure is the largest of roughly sixty correlations computed across five normalisations, three numerators and four candidates, none named in advance. It is a basis for one pre-registration, not a result. The low-rank-at-both-extremes observation rests on two points of ten, the configuration Ф67 warns about.

<!-- marker: repo-state v3 applied -->
