# Experiment 16: Double Pendulum Sweep (4 DOF) — UPDATED

## What this tests
Does prescribed advantage hold on double pendulum (θ₁, ω₁, θ₂, ω₂)?

## Key results — UPDATED 15.04.2026

**Original (raw, no normalization):** prescribed loses at dim ≥ 2.
**With min-max [0,1] normalization:** prescribed WINS at ALL dim 1–8.

| Dim | Presc RAW | Presc NORM | Free | NORM/Free |
|-----|-----------|-----------|------|-----------|
| 1 | 0.052 | **0.001** | 0.014 | **9.8×** |
| 2 | 0.207 | **0.001** | 0.021 | **16.8×** |
| 4 | 0.078 | **0.002** | 0.032 | **19.1×** |
| 8 | 0.078 | **0.002** | 0.026 | **15.4×** |

Normalization gives 37–166× improvement to prescribed encoder.

**This resolves П1:** the difference between Push-T and pendulums was normalization, not environment structure.

## Setup
- **Environment:** Double pendulum (θ₁, ω₁, θ₂, ω₂), synthetic
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200

## Files
```
results/fix_e16_results.json    — Full results with raw, norm, and free (per-seed)
results/results.json            — Original results (scalar, no per-seed, no normalization)
```

## Facts
Ф20 (updated)

## Resolves
П1 (Push-T vs pendulums contradiction) — CLOSED
