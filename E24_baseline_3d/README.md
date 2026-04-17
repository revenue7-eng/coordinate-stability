# Experiment 24: Baseline 3D Comparison

## What this tests
Baseline measurement for Tier 3 high-dimensional experiments, using same architecture as E25-E26.

## Key results
| Metric | Value |
|--------|-------|
| Gap (prescribed/free) | 169× |
| Drift (epoch 0→1) | 1.53 |
| R² transfer | −70 |

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier3_highdim.py           — Complete Tier 3 script (baseline section)
results/tier3_results.json      — Results (5KB, baseline_3d key)
```

## Facts
Included in Ф33
