# Experiment 8: Random Fixed Encoder Control

## What this tests
Does a frozen random orthogonal projection (zero semantic content) outperform a trainable free encoder?

## Key results
| Condition | Val Loss | Ratio |
|-----------|----------|-------|
| Prescribed | 0.000036 | 1× |
| Rotated prescribed | 0.000039 | 1.09× |
| Random fixed 5→3 | 0.000476 | 13× worse than prescribed |
| Free | 0.008282 | 230× worse than prescribed |

- Random fixed 17× better than free (Ф12) — stability without alignment gives order-of-magnitude advantage
- Prescribed 13× better than random fixed (Ф13) — alignment adds on top of stability
- Rotated ≈ prescribed (Ф14) — axis interpretability doesn't matter

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/random_fixed_encoder.py                — Standalone script
code/paper2_random_fixed_v2_colab.ipynb     — Colab notebook
results/random_fixed_v2_results.json        — Results (39KB)
```

## Facts
Ф12, Ф13, Ф14
