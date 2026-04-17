# Experiment 23: Random Projection 3D vs 5D Subspace

## What this tests
Is the prescribed advantage about axis alignment within the subspace, or about subspace selection itself?

## Key results
| Condition | Val Loss |
|-----------|----------|
| Prescribed | ~0.000037 |
| Rotated prescribed | ~0.000037 |
| Random fixed 3D (block coords) | ~0.000037 |
| Random fixed 5D (all coords) | 376,053 (EXPLODES) |
| Free | — |

- random_fixed_3d ≈ prescribed ≈ rotated_prescribed (Ф31) — alignment within subspace is irrelevant
- random_fixed_5d EXPLODES (Ф32) — wrong subspace is catastrophic
- Mechanism = subspace selection + normalization + freeze, not axis orientation

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier2_confound_tests.py    — Complete Tier 2 script (T7 section)
results/tier2_results.json      — Results (3KB, T7 key)
```

## Facts
Ф31, Ф32

## Note
This result invalidates the original E08 decomposition (17× × 13× = 233×). The "17× stability" in E08 was an artifact of a specific random_fixed implementation.
