# Experiment 22: Optimizer State Preservation in Freeze Test

## What this tests
Is the freeze test result (E07) confounded by optimizer reset when encoder parameters are frozen?

## Key results
| Condition | New Optimizer | Keep State | Difference |
|-----------|--------------|-----------|------------|
| Freeze@1 | 0.008785 | 0.008701 | 1.0% |
| Freeze@3 | — | — | 2.9% |

- Optimizer reset is NOT a confound (Ф30)
- Difference < 3% — negligible compared to the 20% freeze effect

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier2_confound_tests.py    — Complete Tier 2 script (T5 section)
results/tier2_results.json      — Results (3KB, T5 key)
```

## Facts
Ф30
