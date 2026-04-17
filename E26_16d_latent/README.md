# Experiment 26: 16D Latent Space

## What this tests
What happens at high dimension with engineered nonlinear features?

## Key results
| Condition | Val Loss | Ratio |
|-----------|----------|-------|
| Prescribed 16D | baseline | 1× |
| Free MLP 5→16 | — | 50× worse |
| Random fixed 5→16 | — | 1.53× worse |

- Gap prescribed/free: 50× (Ф33) — still large at 16D
- Random fixed / prescribed: 1.53× — alignment begins to matter at high dim (Ф34)
- Drift 3.58, R² transfer −596 — drift amplifies with dimension (Ф35)

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier3_highdim.py           — Complete Tier 3 script (T9b section)
results/tier3_results.json      — Results (5KB, T9b_16d key)
```

## Facts
Ф33, Ф34, Ф35
