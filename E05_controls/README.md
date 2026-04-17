# Experiment 5: Controls — Random Fixed, Equal-Input, SIGReg Ablation

## What this tests
Three control experiments to isolate the mechanism behind prescribed axes advantage:
1. Random fixed axes (frozen random projection) vs prescribed
2. Equal-input free encoder (receives same x,y,θ) vs prescribed
3. SIGReg ablation (±SIGReg on free encoder)

## Key results
| Condition | Val Loss | Ratio vs Prescribed |
|-----------|----------|---------------------|
| Prescribed 3D | 0.000472 | 1× |
| Random fixed 3D | 0.000290 | 0.61× (slightly better) |
| Free 3D same input | 0.003570 | 7.6× worse |
| Free −SIGReg | 0.006 | improved |
| Free +SIGReg | 0.012 | 1.9× worse than −SIGReg |

**Conclusions:**
- Fixation > semantics (Ф5): random fixed ≈ prescribed
- Not information access (Ф6): equal-input free 7.6× worse
- SIGReg treats symptoms prescribed prevents (Ф7)

## Setup
- **Environment:** Push-T (gym-pusht, real physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 50
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/           — Controls used same codebase as E03 (lewm_pusht_experiment.py)
results/        — Results published in Paper 1 (Section 4.6, reviewer response)
```

## Facts
Ф5, Ф6, Ф7

## Related
- **exp5a_random_axes_scaling:** Same experiment at 500 episodes — free encoder wins 695,000× (Ф38)
- **exp5b_gauge_fix:** Gauge fixing as alternative to prescribed — does not help (Ф37)

## How to reproduce
Same as E03, with additional conditions: `--condition random_fixed`, `--condition free_same_input`, `--no-sigreg`.
