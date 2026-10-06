# E43: External-target sweep over the E42 encoders (Push-T, gym-pusht)

## What this tests
Whether the E42 encoders keep their ordering when the head is trained on an external target (PE of the true state) instead of the encoder's own latent, and which pre-registered predictor orders them there.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф75: E43, on an external target the spread shrinks and R2_readout orders the frozen encoders
- [verified: python3 E43_external_target/code/analyze_e43.py, 2026-09-28; pre-registration 410679a] Data seed 42, 30 cells, the same frozen encoders as E42 (enc_at_freeze tensor-equal to the E42 checkpoint in every cell), each head trained on PE of the state at window position 3. final_vp spans 1.7425x with sd(log) 0.1549, against sd(log) 0.3986 for E42 final_vp on the same encoders: R = 0.3887. Data seed 123, 10 cells: R = 0.2984.
- [verified: same] Noise floor: sd(log final_vp) over the six heads on encoder 1 is 0.0161, F = 0.1039.
- [verified: same] Registered statistic, Pearson against log(final_vp) with two-sided Fisher p and threshold 0.0167: log(persistence) -0.0779 (p 0.68498; predicted no association), R2_readout -0.8593 (p < 0.00001; predicted negative; hit), eff_rank -0.0675 (p 0.72533; predicted negative; no hit). Spearman, reported and deciding nothing: -0.0590, -0.8180, -0.0790.
- [verified: same] Jackknife ranges: log(persistence) [-0.1953, +0.0208], R2_readout [-0.8775, -0.8424], eff_rank [-0.1954, +0.0089]. No single point carries the R2_readout figure.
- [verified: same] Ordering agreement with E42 final_vp over the 30 encoders: Spearman +0.0171.
- STATUS of the verdict: analyze_e43.py prints COLLAPSE (R < 0.5 and predictor 1 not a hit). The declared outcomes evaluated predictors 2 and 3 only under R >= 0.5 and named no outcome for a collapse together with a hit on predictor 2. The R2_readout hit is fully registered (statistic, threshold and sign fixed in 410679a). The joint reading below is interpretation, not a declared outcome.
- CONSEQUENCE, interpretation: most of the E42 spread came from the self-referential target. On a target the encoders do not define, persistence no longer orders them and the E42 ordering does not survive. A smaller spread remains, about ten times the head-noise floor in sd(log), and linear extractability of the true state, measurable before training, accounts for about 74% of its variance (r^2 = 0.738).
- CORRECTS Ф64: its null (corr +0.060 at n=10, and +0.0235 at n=30 in Ф69) is a property of the self-referential metric, not evidence that linear informativeness fails to order frozen encoders.
- CORRECTS Ф74: the eff_rank lead (-0.4088 against the two-predictor residual) does not carry over to the external target (-0.0675, p 0.725).
- NEXT: a confirmatory test of R2_readout alone at n >= 30 on data seed 123, pre-registered separately with one statistic and a predicted sign.

## Setup
From EXPERIMENTS.md, section E43.

- **Status:** PLANNED. Written 2026-09-28 after Ф69 to Ф74 and before any
  external-target cell was run.
- **Question:** is the spread of E42 a property of the frozen encoders, or of
  the self-referential target? In E40 to E42 the head predicts emb[:, 3], the
  encoder's own output, so the encoder sets the scale of the loss it is scored
  by, and persistence orders the initialisations at +0.97 (Ф73). The declared
  E42 outcome "the property that orders frozen bases is not a cheap one" does
  not hold for that metric (Ф73).
- **Design:** the same frozen encoders as E42. Initialisation k at a data seed
  is built with enc_seed=k, head_seed=None, and every cell asserts that its
  frozen encoder equals the E42 checkpoint of the same cell tensor for tensor.
  The head is trained to predict PE(state at window position 3): block x/512,
  block y/512 and block angle/(2*pi) of the fourth state in the window, a fixed
  function of the true state that does not depend on the encoder. Context,
  action encoder, predictor, optimiser, epochs, split and batch order are those
  of E42. Code: `e43_lib.py` is `e42_lib.py` byte for byte plus an appended
  block that rebinds M; the first N bytes hash to c41d9434a5fc0ffc..., the
  sha256 of `e42_lib.py`.
- **Acceptance, already run:** in target "self", enc_seed=1 at data seed 42
  reproduced E40 init 1 bit for bit (best_vp 0.0031448905217346915,
  2026-09-28). That run produces no external-target output.
- **Cells:** 30 at data seed 42 (primary). 10 at data seed 123, which test
  reproduction of the spread and no correlation. 5 noise-floor cells: encoder 1
  at data seed 42 with head_seed 1..5.
- **Metric:** final_vp on the external target, on the log scale. Not best_vp,
  which carries selection over epochs (Ф74).

**Predictors, closed list, copied from E42 for the same encoder (Ф69, Ф73):**
1. log(persistence). The self-reference reading (Ф73) predicts no association:
   persistence ordered E42 through the scale of a target that is absent here.
2. R2_readout, linear extractability of the block pose from the frozen
   representation, which is the quantity the external head is asked to
   recover. The information reading predicts a negative association.
3. eff_rank. Predicted negative, from Ф74, which recorded it as the basis for
   this pre-registration.

Excluded, with reasons: cond_number correlates -0.7810 with eff_rank (Ф74);
smoothness is a function of rms_norm and persistence (Ф72); rms_norm was a
scale control in E42 and the external target has a fixed scale.

**Statistic, one, fixed before the run:**
- Pearson correlation of each predictor with log(final_vp) over the 30 cells at
  data seed 42; two-sided p from Fisher z. A hit is p < 0.0167 (Bonferroni over
  three). The sign is reported against the prediction; a hit with the
  unpredicted sign is recorded as a hit against the prediction. Spearman is
  printed alongside and decides nothing. At n=30 and power 0.80 the threshold
  resolves |r| from 0.553.
- Leave-one-out jackknife for every predictor, significant or not.
- The composition of points is not changed after seeing the results.
- The analysis is `E43_external_target/code/analyze_e43.py`, committed with
  this block. It prints the verdict below mechanically.

**Descriptive quantities with declared thresholds, outside the correction:**
- R = sd(log final_vp, E43) / sd(log final_vp, E42) over the same 30 encoders.
- Ordering agreement: Spearman between E43 final_vp and E42 final_vp over the 30.
- Noise floor F = sd(log final_vp) over the six heads on encoder 1 (head_seed
  None and 1..5) divided by sd(log final_vp, E43) over the 30 encoders.
- R at data seed 123 over its 10 encoders.

**Declared outcomes:**
- F >= 0.5: head noise is comparable to the encoder effect on this target.
  Correlations are reported and not interpreted, whatever their p.
- Collapse: R < 0.5 and predictor 1 is not a hit. The E42 spread is mainly a
  property of the self-referential target: comparing frozen encoders by a loss
  on their own output ranks them by latent step size.
- Persistence of the spread: R >= 0.5. The encoders differ on a target they do
  not define. A hit on predictor 2 or 3 with the predicted sign becomes a lead,
  to be confirmed in a separate experiment at n >= 30 on data seed 123. No hit
  means the difference is real and none of the three orders it.
- Predictor 1 a hit: persistence carries information about the encoder beyond
  the scale of the self-referential target. Recorded whatever R is.
- R at data seed 123 is reported next to R at data seed 42 and tests
  reproduction only.

**Cost:** E42 cells took 74.2 to 128.1 s on an unloaded machine (ce98ae4); the
E43 acceptance cell took 141 s. 45 cells: 60 to 106 min. Start only with no
concurrent build: cc1plus count 0 and load average under 1.

## Files
- `code/__pycache__`
- `code/analyze_e43.py`
- `code/e43_lib.py`
- `code/run_e43.py`
- `results/run_e43.log`
- `results/sweep.json`

## Facts
Ф75

## How to reproduce
```
python run_e43.py probe      acceptance only, target "self"
python run_e43.py            data seed 42, external target, resumable
python run_e43.py second     data seed 123, external target, resumable
python run_e43.py floor      encoder 1 at data seed 42, head_seed 1..5,
external target, resumable
python3 E43_external_target/code/analyze_e43.py
```

## Status
Summary table: R2_readout orders on an external target.

- Ф75: STATUS of the verdict: analyze_e43.py prints COLLAPSE (R < 0.5 and predictor 1 not a hit). The declared outcomes evaluated predictors 2 and 3 only under R >= 0.5 and named no outcome for a collapse together with a hit on predictor 2. The R2_readout hit is fully registered (statistic, threshold and sign fixed in 410679a). The joint reading below is interpretation, not a declared outcome.

<!-- marker: repo-state v3 applied -->
