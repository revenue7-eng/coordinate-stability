# Experiment 28: Dim Sweep Full Parameters (П2 Resolution)

## What this tests
Repeats E13 (dim sweep) with full Tier 3 parameters to resolve П2 contradiction.

## Key results

| Dim | Prescribed | Free | Ratio | Winner |
|-----|-----------|------|-------|--------|
| 1 | 0.000037 | 0.002239 | 60× | PRESCRIBED |
| 2 | 0.000005 | 0.008481 | 1820× | PRESCRIBED |
| 3 | 0.000036 | 0.008282 | 228× | PRESCRIBED |
| 4 | 0.000161 | 0.018366 | 114× | PRESCRIBED |
| 5 | 0.000361 | 0.023927 | 66× | PRESCRIBED |
| 7 | 0.000494 | 0.028016 | 57× | PRESCRIBED |
| 11 | 0.000732 | 0.030509 | 42× | PRESCRIBED |

**NO CROSSOVER.** Prescribed wins at all dimensions 1–11.

Old E13 (100ep, 20 epochs, 2 seeds) showed crossover at dim=4 — this was an artifact of underpowered setup.

dim=5 matches Tier 3 E25 exactly: 66.3× vs 66.2×.

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Predictor:** hidden = max(128, dim × 8)

## Facts
Ф17 (updated), Ф18 (refuted)

## Resolves
П2 (dim sweep vs Tier 3 contradiction) — CLOSED
