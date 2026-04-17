# Experiment 19: Update Ratio + Differential LR

## What this tests
Is the prescribed advantage purely an optimization lag issue (encoder updates too fast relative to predictor)?

## Key results
| Condition | Val Loss | Ratio vs Prescribed |
|-----------|----------|---------------------|
| Prescribed | baseline | 1× |
| Free K=1 (baseline) | — | 222× |
| Free diffLR 100× | — | 62× |
| Free K=3 | — | worse than K=1 (−26%) |

- DiffLR 100× reduces gap from 222× to 62× — 72% improvement, but 62× remains (Ф26)
- Extra predictor steps (K=3) make things WORSE
- NOT pure optimization lag

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier1_all_tests.py     — Complete Tier 1 script (T2 section)
results/tier1_results.json  — Results (30KB, T2 key)
```

## Facts
Ф26
