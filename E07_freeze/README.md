# Experiment 7: Freeze Test

## What this tests
If drift causes degradation, then freezing the free encoder early should help.

## Key results
| Freeze point | Best val loss | Δ vs unfrozen |
|-------------|--------------|---------------|
| Unfrozen | 0.081 | baseline |
| Freeze@1 | 0.065 | +20% |
| Freeze@2 | — | +2.9% |
| Freeze@10 | — | −1.1% |

- Freeze@1 improves by 20% — causal evidence for drift harm (Ф11)
- But freeze@1 (0.065) still 25× worse than prescribed (0.0025)
- Stability alone is not sufficient

## Setup
- **Environment:** Push-T (gym-pusht, real physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/freeze_test_standalone.py  — Standalone freeze test script
results/all_results.json        — Shared results file with E06
```

## Facts
Ф11

## How to reproduce
```bash
python code/freeze_test_standalone.py
```
