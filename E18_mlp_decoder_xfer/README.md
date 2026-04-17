# Experiment 18: MLP Decoder Transfer

## What this tests
Does early drift destroy information irreversibly, or just make it linearly unreadable?

## Key results
| Transition | MLP R² Transfer | Linear R² Transfer |
|-----------|----------------|-------------------|
| Epoch 0→1 | −283 | −71 |
| Epoch 5→6 | 0.79 | 0.69 |

- Early drift (ep 0→1) destroys information — even MLP decoder breaks (Ф24)
- Late drift (ep 2+) preserves information in nonlinearly transformed form (Ф25)
- Two-phase drift model confirmed

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier1_all_tests.py     — Complete Tier 1 script (T1 section)
results/tier1_results.json  — Results (30KB, T1 key)
```

## Facts
Ф24, Ф25
