# Experiment 14: Lower Boundary (dim 1–3)

## What this tests
How does the prescribed advantage scale at very low dimensions?

## Key results
| Dim | Prescribed/Free Ratio |
|-----|----------------------|
| 1 | 78× |
| 2 | 12× |
| 3 | 1.5× |

- Maximum advantage at minimum dimension
- Monotonic decrease — each added dimension reduces the gap

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 2–3 seeds
- **Epochs:** 15–20
- **Episodes:** 50–100
- **Platform:** Google Colab T4 GPU

## Files
```
code/run_lower.py       — Experiment script
results/output.txt      — Raw output (<1KB)
```

## Facts
Included in Ф18

## Data provenance
Only output.txt from console. No JSON, no per-seed data.
Numbers verified against output.txt but not independently reproducible from saved data.
