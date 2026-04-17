# Experiment 27: Drift Rate vs Downstream Quality Correlation

## What this tests
Is there a quantitative relationship between drift rate and downstream prediction quality?

## Key results
| Metric | Value |
|--------|-------|
| Pearson correlation | 0.95 |
| Spearman correlation | 0.51 (nonlinear relationship) |

- Two regimes: catastrophe (drift > 0.3) and saturation (drift < 0.1) (Ф28)
- R² ceiling ≈ 0.75 (linear decoder on free encoder)
- Early drift 8–14× larger than late drift

## Setup
- **Environment:** Push-T (gym-pusht, real physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Data source:** Reanalysis of E06 data (no new training)
- **Platform:** Google Colab T4 GPU

## Files
```
code/                       — Analysis performed in-conversation (no standalone script)
results/all_results.json    — Data from E06 (142KB)
```

## Facts
Ф28

## Note
This is an analysis experiment — no new training runs. Uses existing data from E06.
