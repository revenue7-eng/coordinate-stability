# Experiment 21: Aligned-Drifting ± SIGReg

## What this tests
Does SIGReg destroy aligned initialization, or does it stabilize it?

## Key results
| Condition | Val Loss |
|-----------|----------|
| Prescribed | baseline |
| Aligned-linear + SIGReg | 356× worse |
| Aligned-linear − SIGReg | 12601× worse (catastrophic on seed 123) |
| Free + SIGReg | 0.008 |
| Free − SIGReg | 0.007 (slightly better) |

- SIGReg stabilizes aligned-linear — prevents divergence on seed 123 (Ф29)
- Neither variant approaches prescribed
- Hypothesis Г13 refuted: SIGReg stabilizes, not destroys

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier2_confound_tests.py    — Complete Tier 2 script (T4 section)
results/tier2_results.json      — Results (3KB, T4 key)
```

## Facts
Ф29
