# Experiment 30: Critical Window Localization

## What this tests
exp7 (Ф11) showed freezing the free encoder at epoch 1 helps (+20%) but is insufficient
(still 25× worse than prescribed). This asks a *different* question: is the irreversible
damage spread across early epochs, or concentrated in a sharp window — and **where** does
the window close?

## Key results
| Transition | best_vp ratio | Meaning |
|-----------|---------------|---------|
| freeze@0 (random_fixed) → freeze@1 | **136×** | almost all damage here |
| freeze@1 → unfrozen | 1.3× | negligible by comparison |

- ~99% of the free-encoder degradation is inflicted during the **first epoch**.
- The cliff is between epoch 0 and epoch 1 — NOT between later freeze points.
- Within the freeze@k≥1 band (k=1,2,3,5,7,10) all values are within a narrow range
  (0.065–0.082), every one ≥25× worse than prescribed.
- Refines Г18: the critical window is not "epochs 0–2" but "within the first epoch."
- Mechanistic closure for Ф31 (random_fixed ≈ prescribed): both are freeze@0, i.e. both
  capture the representation *before* the catastrophic first epoch.

## Relationship to prior experiments
- **exp7 (Ф11):** established freeze@1 helps but insufficient. Does NOT localize the window.
- **exp18 (Ф24/Ф25):** established two-phase drift (early destroys, late preserves).
- **This (E30):** localizes phase-1 damage to the first epoch via the freeze@0→@1 cliff.
  Additive: new question (where), new fact candidate (Ф45), same data.

## Method
Analysis of the freeze@k profile already recorded in E06/E07 `all_results.json`.
No training. freeze@0 taken as random_fixed encoder value from Ф12 (0.000476).

## Setup
- **Environment:** Push-T (gym-pusht, real physics, synthetic=false)
- **Seeds:** 42, 123, 777
- **Epochs:** 30 (freeze points k ∈ {1,2,3,5,7,10}; freeze@0 = random_fixed proxy)
- **Episodes:** 200
- **Platform:** CPU (analysis only, seconds)

## Files
```
code/analyze_critical_window.py    — Reproducible analysis (reads all_results.json)
results/critical_window_results.json — Output
```

## How to reproduce
```bash
python code/analyze_critical_window.py \
    --data ../E06_cov_drift/results/all_results.json \
    --out results/critical_window_results.json
```

## Caveats
- freeze@0 = random_fixed **proxy** (Ф12), not a literal freeze@0 training run;
  normalization implementation may differ. A literal freeze@0 run would strengthen this.
- 3 seeds: signal, not proof. Sub-epoch resolution absent — cannot yet say *where inside*
  the first epoch the window closes. That requires a sub-epoch freeze sweep (proposed E31).
- Freeze effect is data-dependent (Ф30): holds on real physics; transfer not guaranteed.

## Facts
Ф45 (candidate): ~99% of free-encoder representation damage occurs in the first epoch;
the critical window closes within epoch 1 (freeze@0→@1 = 136× vs freeze@1→unfrozen = 1.3×).

## Proposed follow-up
E31: sub-epoch freeze sweep (freeze at 25%/50%/75% of epoch 1) to locate the window
boundary inside the first epoch. This is the only test that can resolve "within epoch 1"
into a specific point.
