# Experiment 5a: Random Axes Scaling — 200 vs 500 Episodes

## What this tests
Does the prescribed axes advantage persist with more data, or is it a low-data phenomenon?

## Key results

**Low-data (200 episodes, 30 epochs, 3 seeds, no SIGReg):**

| Condition | Val Loss | vs Prescribed |
|-----------|----------|---------------|
| prescribed | 0.000673 | 1× |
| random_fixed | 0.000408 | 0.61× (better) |
| free_3d | 0.003007 | 4.47× worse |

**High-data (500 episodes, 50 epochs, 3–9 runs, no SIGReg):**

| Condition | Val Loss | vs Prescribed |
|-----------|----------|---------------|
| prescribed | 8.513×10⁻⁴ | 1× |
| random_fixed | 8.514×10⁻⁴ | 1.00× |
| free_3d | 1.225×10⁻⁹ | **695,000× better** |
| free_5d | 2.916×10⁻⁹ | 292,000× better |

**Isotropic normalization control (200 ep):**

| Condition | Val Loss | vs non-iso |
|-----------|----------|------------|
| prescribed_iso | 0.010250 | 15.2× worse |
| random_fixed_iso | 0.007019 | 17.2× worse |

**Conclusions:**
- At 500 episodes, free encoder converges to ~10⁻⁹ while fixed encoders plateau at ~8.5×10⁻⁴
- Prescribed axes = sample efficiency, not absolute superiority
- Random ≈ prescribed at both scales (axis semantics secondary)
- Min-max [0,1] normalization critical; standardization 15× worse

## Setup
- **Environment:** Push-T (synthetic physics)
- **200ep:** 3 seeds (42, 123, 777), 30 epochs
- **500ep:** 3 training seeds × 3 rotation seeds = 9 runs for random_fixed, 3 runs for others, 50 epochs
- **No SIGReg** — pure MSE prediction loss

## Files
```
code/run_random_axes_control.py     — 200ep experiment
code/run_isotropic_control.py       — isotropic normalization control
results/all_results_500ep.json      — 18 runs with training curves (500ep)
results/random_fixed_200ep.json     — 200ep results
```

## Facts
Ф38, Ф39, Ф40

## Verification
All numbers verified against JSON: all_results_500ep.json (18 runs, per-run best_val_loss).
