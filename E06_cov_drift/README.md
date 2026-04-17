# Experiment 6: Covariance + Drift Analysis

## What this tests
Does the free encoder degrade because of rank collapse, or because of coordinate drift?

## Key results
| Metric | Free | Prescribed |
|--------|------|-----------|
| Effective rank | 2.99/3 | 2.91/3 |
| Isotropy | 0.86 | 0.66 |
| Val loss | 233× worse | baseline |
| R² transfer (ep 0→1) | −16.9 / −62.2 / −25.4 | ~1.0 |

- Free has full rank and high isotropy — yet loses 233× (Ф9)
- R² transfer catastrophically negative after one epoch (Ф10)
- 80% of drift is structural (persists after Procrustes alignment)
- SIGReg harms free: 4.2× worse (Ф8)

## Setup
- **Environment:** Push-T (gym-pusht, real physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **SIGReg:** λ=0.09
- **Platform:** Google Colab T4 GPU

## Files
```
code/paper2_full_analysis.py            — Full analysis script
code/drift_analysis_standalone.py       — Drift metrics standalone
code/covariance_analysis_standalone.py  — Covariance analysis standalone
code/paper2_colab.ipynb                 — Colab notebook
results/all_results.json                — Complete results (142KB)
```

## Facts
Ф8, Ф9, Ф10

## How to reproduce
```bash
pip install gym-pusht torch numpy
python code/paper2_full_analysis.py
```
