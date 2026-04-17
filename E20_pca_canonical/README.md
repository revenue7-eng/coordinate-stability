# Experiment 20: PCA Canonicalization

## What this tests
If drift is just rotation/scaling, PCA alignment should fix it.

## Key results
- PCA worsens R² transfer at most epochs (Ф27)
- Drift is nonlinear — not rotation or scaling
- PCA canonicalization cannot solve the drift problem

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier1_all_tests.py     — Complete Tier 1 script (T3 section)
results/tier1_results.json  — Results (30KB, T3 key)
```

## Facts
Ф27
