# Experiment 10: Learning Rate Sweep + EMA Baseline

## What this tests
Can optimization tuning (lower LR) or EMA stabilization close the gap with prescribed?

## Key results
| Condition | Best Val Loss | Ratio vs Prescribed |
|-----------|--------------|---------------------|
| Prescribed | baseline | 1× |
| Free LR=1e-4 | — | 4.3× |
| Free LR=3e-4 | — | 5.1× |
| Free LR=1e-3 | — | 6.2× |
| Free LR=3e-3 | — | 7.0× |
| Free + EMA (decay=0.996) | — | 6.1× |

- Prescribed wins at every learning rate (4.3–7.0×)
- EMA 6.1× worse than prescribed — stabilization through averaging doesn't work
- Not an optimization problem

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seed:** 42
- **Epochs:** 50
- **Platform:** Google Colab T4 GPU

## Files
```
code/lr_sweep_ema_baseline.ipynb    — Colab notebook
results/lr_sweep_results.json       — Results (104KB)
```

## Facts
Paper 2, Section 5.6–5.7
