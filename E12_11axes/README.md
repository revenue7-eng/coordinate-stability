# Experiment 12: 11 Prescribed Axes

## What this tests
Does increasing prescribed dimensions to 11 (inspired by M-theory) improve performance?

## Key results
| Condition | Val Loss | Ratio vs Prescribed_3 |
|-----------|----------|----------------------|
| prescribed_3 | 0.000019 | 1× |
| prescribed_11 | 0.000381 | 20× worse |
| free_3 | — | — |
| free_11 | 0.000060 | 3× worse (but 6× better than prescribed_11) |
| random_fixed_11 | — | — |

- Prescribed_11 is 20× worse than prescribed_3 (Ф17)
- Free_11 beats prescribed_11 by 6×
- Redundant fixed axes harm performance — hypothesis Г6 refuted

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/run_11axes.py          — Experiment script
results/results.json        — Results (2KB)
```

## Facts
Ф17
