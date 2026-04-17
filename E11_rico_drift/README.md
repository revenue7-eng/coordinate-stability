# Experiment 11: Rico UI Drift Analysis (Vision)

## What this tests
Does drift occur in vision modality (UI screenshots), confirming cross-modal generality?

## Key results
| Metric | Free | Prescribed |
|--------|------|-----------|
| R² transfer | 0.93 | ~1.0 |
| Effective rank | measured | measured |
| Condition number | measured | measured |

- Drift in vision weaker than Push-T (R² 0.93 vs 0.78) but present
- Cross-modal confirmation of drift phenomenon

## Setup
- **Environment:** Rico dataset, 398 UI screenshots
- **Seed:** 42
- **Epochs:** 100
- **Platform:** Google Colab T4 GPU

## Files
```
code/rico_drift_v2.ipynb                — Colab notebook
results/rico_drift_v2_results.json      — Results (53KB)
```

## Facts
Paper 2, Section 5.8
