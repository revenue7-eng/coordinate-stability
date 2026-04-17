# Experiment 5b: Gauge Fixing Free Encoder

## What this tests
Can fixing the gauge symmetry (rotation/reflection invariance) of the free encoder improve its performance? This is an alternative to prescribed axes: instead of fixing the coordinates, fix the symmetry.

## Key results

| Condition | Val Loss | vs Prescribed |
|-----------|----------|---------------|
| prescribed | 0.000472 | 1× |
| free | 0.011614 | 24.6× |
| gauge_fixed_free | 0.012581 | 26.7× |
| linear_free | 16,009 | exploded |

**Conclusion:** Gauge fixing does NOT help — 1.08× ≈ free. The problem is not gauge symmetry but coordinate drift. Linear free encoder explodes completely.

## Setup
- **Environment:** Push-T (synthetic physics)
- **Data seed:** 42
- **Training seeds:** 42, 123, 777
- **Epochs:** 50
- **Date:** 12 April 2026

## Files
```
results/results.json    — Full results with per-seed breakdown and ratios
```

## Facts
Ф37

## Verification
Numbers verified from results.json: gauge_fixed_over_free = 1.083×.

## Note
This experiment was conducted on 12 April 2026 but was not included in FACTS.md until the audit on 15 April 2026. Added as Ф37.
