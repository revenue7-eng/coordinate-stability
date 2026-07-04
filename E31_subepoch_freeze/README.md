# Experiment 31: Sub-Epoch Freeze Sweep

## What this tests
E30 localized the free-encoder damage to the FIRST epoch. This asks: inside epoch 1, does
damage appear at a SHARP threshold (a discrete moment of no-return) or accumulate as a
SLOPE (gradual erosion)? Encoder is frozen at fractions of epoch-1 batches
(0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0); best_vp vs freeze fraction gives the shape.

## Key results
| freeze @ (frac of epoch 1) | best_vp (mean, 5 seeds) |
|---|---|
| 0.00 | 0.00165 |
| 0.25 | 0.00223 |
| 0.30 | 0.00278 |
| 0.35 | 0.00358 |
| 0.40 | 0.00467 |
| 0.45 | 0.00593 |
| 0.50 | 0.00775 |
| 0.60 | 0.01245 |
| 1.00 | 0.01736 |

- **Verdict: SLOPE, not threshold.** Over the rise (f≥0.25) a linear fit beats the best
  single step by 2.2× in residual (SS 0.085 vs 0.186); linear R²=0.88 in log space.
- Monotone in 3/5 seeds with zero downward steps; the other 2 have a single noise dip each.
  No seed shows a sharp jump.
- First quarter of epoch 1 (≤ freeze@0.25) is near-harmless; damage then accumulates
  continuously, accelerating toward epoch end.
- Refines Г18/E30: the critical window is not a wall but a slope — damage integrates over
  training steps within epoch 1, consistent with continuous drift rather than a discrete event.

## Setup
- **Environment:** Push-T (synthetic physics via synth())
- **Seeds:** 42, 123, 777, 7, 99
- **Epochs:** 20   **Episodes:** 100   (moderate; sufficient for a within-run shape question)
- **Fractions:** 0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0
- **Platform:** CPU, ~45 runs × ~90–240s ≈ 60–70 min

## Design notes
- The shape question is a WITHIN-run comparison, so it does not need exp7's absolute scale
  (200ep/30epoch); compute is spent on grid density and seeds instead.
- Pilot (5 fracs, 2 seeds) was INCONCLUSIVE (width CI straddled). v2 (dense grid, 5 seeds)
  resolves it.
- Primary shape test is linear-vs-best-step residual over the rise — independent of any
  threshold calibration. A width-based CI is reported as secondary and is not load-bearing.

## Files
```
code/subepoch_freeze.py            — run (resume-safe: saves per seed, skips completed)
code/analyze_shape.py              — shape verdict (linear-vs-step primary; width CI secondary)
results/subepoch_freeze_results.json — profile (5 seeds)
results/shape_verdict.txt          — analyzer output (verdict: SLOPE)
```

## Facts
Ф46 (candidate): inside epoch 1, free-encoder representation damage accumulates as a
SLOPE, not a threshold — linear in log(best_vp) over the rise (R²=0.88), the best single
step fitting 2.2× worse. First ~quarter of epoch 1 is near-harmless; damage then integrates
continuously. Consistent with continuous drift, against a discrete no-return event.

## Caveats
- Synthetic data (synth()), not real physics (exp6). Freeze effect is data-dependent (Ф30),
  so the SLOPE verdict must be re-checked on gym-pusht data before being taken as firm.
- 2/5 seeds show a single noise dip; on real (noisier) data the monotone trend may be
  less clean, though the linear-over-step preference is large (2.2×).

## How to reproduce
```
python code/subepoch_freeze.py --episodes 100 --epochs 20 \
    --seeds 42 123 777 7 99 \
    --fracs 0.0 0.25 0.30 0.35 0.40 0.45 0.50 0.60 1.0 \
    --out results/subepoch_freeze_results.json
python code/analyze_shape.py --data results/subepoch_freeze_results.json
```

## Extends
exp7_freeze (Ф11), exp30_critical_window (Ф45 candidate).

## Proposed follow-up
E32: same sweep on real gym-pusht data (exp6 pipeline) to remove the synthetic caveat and
confirm SLOPE under Ф30 data-dependence.
