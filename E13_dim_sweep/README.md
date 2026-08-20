# Experiment 13: Dimension Sweep 3–15

## What this tests
At which latent dimension does free overtake prescribed?

## Key results
| Dim | Prescribed | Free | Winner |
|-----|-----------|------|--------|
| 3 | best | — | prescribed (1.5×) |
| 4 | — | — | free (crossover) |
| 5–15 | — | — | free |

- Crossover at dim=3→4 (Ф18)
- Monotonic degradation of prescribed advantage with dimension
- Maximum advantage at minimum dimension

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 3 seeds
- **Epochs:** 20
- **Episodes:** 100 (preliminary)
- **Platform:** Google Colab T4 GPU

## Files
```
code/run_sweep.py               — Sweep script
results/sweep_results.json      — Results (3KB)
```

## Facts
Ф18

## Note
Preliminary data (100 episodes, 20 epochs). Contradicts E25 (prescribed_5d wins 66×) — see П2 in EVIDENCE.md.

## Data provenance
JSON contains only 2 seeds (42, 123). README mentions 3 seeds — third seed missing from data.
Being superseded by p2_dim_sweep_full.py (E28) with 3 seeds × 200ep × 30 epochs.
