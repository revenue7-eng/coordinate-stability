# Experiment 9: Aligned-but-Drifting + 2×2 Factorial

## What this tests
If an encoder starts perfectly aligned with ground truth but is allowed to drift, does it retain its advantage?

## Key results
| Condition | Val Loss |
|-----------|----------|
| Prescribed (stable+aligned) | 0.000039 |
| Random fixed (stable+unaligned) | 0.000473 |
| Aligned-drifting linear (unstable+aligned) | 0.012849 |
| Free (unstable+unaligned) | 0.008282 |

- Aligned-drifting ≈ free or worse (Ф15) — alignment without stability is useless
- 2×2 factorial interaction: stability × alignment = 19× (Ф16) — hierarchical, not independent
- Stability is a prerequisite; alignment adds 12× only when stability is present

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/paper2_aligned_drifting_colab.ipynb    — Colab notebook
results/aligned_drifting_results.json       — Results (49KB)
```

## Facts
Ф15, Ф16
