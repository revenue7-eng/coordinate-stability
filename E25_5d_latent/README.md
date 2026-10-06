# Experiment 25: 5D Latent Space

## What this tests
Does prescribed advantage hold when ALL 5 state coordinates are prescribed (no information selection)?

## Key results
| Condition | Val Loss | Ratio |
|-----------|----------|-------|
| Prescribed 5D | baseline | 1× |
| Free MLP 5→5 | — | 66× worse |
| Random fixed 5→5 | — | 0.92× (≈ prescribed) |

- Gap prescribed/free: 66× (Ф33) — prescribed works even without subspace selection
- Random fixed ≈ prescribed (Ф34) — stability still dominates
- Prescribed without subspace selection still works (Ф36)
- Drift 1.91, R² transfer −65

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/tier3_highdim.py           — Complete Tier 3 script (T9a section)
results/tier3_results.json      — Results (5KB, T9a_5d key)
```

## Facts
Ф33, Ф34, Ф36

## Significance
Critical result — contradicts E13 dim sweep (crossover at dim=4) and challenges the information selection hypothesis. See П2 in EVIDENCE.md.

## Status
Summary table: Ф36 corrected by Ф78.

- Ф33: STATUS: not interpretable as quality (own latent, Ф76 free encoder); see Ф88.
- Ф34: STATUS: 5D rotation equality, 16D different geometry; see Ф88.
- Ф36: STATUS at dim 5: corrected by Ф78; E25 shares the E28 encoders and metric, see Ф80.; STATUS: not interpretable as a quality gap between encoders until both are scored on a common target with equally scaled input; not refuted. See Ф77.
