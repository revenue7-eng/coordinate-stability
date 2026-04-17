# Experiment 17: Fragility Test — 4th Axis Types

## What this tests
How sensitive is prescribed to adding a single additional axis? What types of 4th axes cause what damage?

## Key results
| Condition | Val Loss | Ratio vs prescribed_3 |
|-----------|----------|----------------------|
| prescribed_3 (x,y,θ) | 0.000019 | 1× |
| +sin(θ) (same subspace) | 0.000093 | 4.8× |
| +agent_x (cross-subspace) | 0.000154 | 7.9× |
| +distance (cross-subspace) | 0.000160 | 8.2× |
| +noise (unpredictable) | 0.021534 | 1106× |

- Unpredictable axis is catastrophic: 1106× (Ф21)
- Same-subspace axes less harmful than cross-subspace (Ф22)
- Redundancy vs independence doesn't matter within cross-subspace: dist ≈ agent_x (Ф23)

## Setup
- **Environment:** Push-T (synthetic physics)
- **Seeds:** 42, 123, 777
- **Epochs:** 30
- **Episodes:** 200
- **Platform:** Google Colab T4 GPU

## Files
```
code/run_fragility.py           — Experiment script
results/results.json            — Results (1KB)
```

## Facts
Ф21, Ф22, Ф23
