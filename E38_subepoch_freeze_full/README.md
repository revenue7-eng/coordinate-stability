# E38 — Sub-epoch freeze sweep, full budget + sub-0.25 resolution

**Date:** 2026-09-11 · **Branch:** drift / hallucination (Paper 2 line)
**Purpose:** Close the three open items E32 left behind: the forced compute budget
(EP=4, NEP=50), the missing resolution below f=0.25, and the absent per-seed raw
data. Same design, same infrastructure, same analyzer.

---

## What this changes about Ф46

Ф46 said: within epoch 1 the damage accumulates as a SLOPE, not a threshold; the
first quarter of epoch 1 is near-harmless; damage integrates continuously,
accelerating toward the end of the epoch.

Measured at full budget with points below 0.25:

[FACT] The SLOPE verdict holds, and holds harder than before. Over the actual rise
(f∈[0.00,0.40]) **5/5 seeds are strictly monotone, zero downward steps**; pooled
linear R²=0.880 and the best single-breakpoint step fit is 1.86× worse than the
line. E31 (synthetic) had 3/5 monotone; E32 (real, reduced budget) had 5/5 over its
own band.

[FACT] **The onset is NOT near-harmless.** From f=0.00 to f=0.25 best_vp rises by
5.8×, 9.8×, 11.0×, 11.2×, 17.1× (seeds 7, 123, 42, 2024, 777). The
"first quarter near-harmless" reading in Ф46/E31 was an artifact of the grid: there
were no points between 0.00 and 0.25, and the two endpoints were joined by a line.

[FACT] **The curve saturates rather than accelerating.** Per-seed plateau onset
(first f from which the curve stays within 5% of its maximum): f=0.70 (seed 42),
0.80 (2024), 0.90 (7), 1.00 (123); seed 777 never satisfies the 5% criterion
because it oscillates near its maximum from f≈0.35 onward. The convexity noted in
E31/E32 was the visible part of a rise whose upper half was outside the measured
band.

[INFERENCE] Corrected shape statement: within epoch 1 the damage rises
continuously from the very first optimizer steps, is steepest over roughly the
first 40% of the epoch, and then saturates. Not a wall, not a uniformly
accelerating ramp: a rise with a plateau, whose knee position varies by seed.

---

## Window sensitivity (why the legacy analyzer now says STEP)

Running E32's `analyze_shape.py` unmodified on this data returns **STEP**
(ratio 0.66, R²_linear 0.576, 1/5 monotone). This is a property of the band, not
of the curve. The 0.25–0.60 band was chosen when 0.25 was the start of the rise;
at full budget that band straddles the rise and the plateau, and two constants fit
a knee better than one line does.

`code/analyze_shape_windows.py` runs the identical comparison over several windows:

| window | pts | monotone | R²_linear | SS_step/SS_lin | verdict |
|---|---|---|---|---|---|
| 0.25–0.60 (legacy E31/E32 band) | 8 | 1/5 | 0.576 | 0.66 | STEP |
| 0.00–0.40 (the actual rise) | 9 | **5/5** | 0.880 | 1.86 | **SLOPE** |
| 0.00–0.45 | 10 | 3/5 | 0.881 | 1.82 | SLOPE |
| 0.00–1.00 (whole curve) | 17 | 0/5 | 0.733 | 0.88 | INCONCLUSIVE |
| 0.45–1.00 (plateau) | 8 | 0/5 | 0.452 | 0.90 | INCONCLUSIVE |

[INFERENCE] SLOPE appears on the rise, STEP only on a band containing the knee,
and nothing is resolvable on the plateau (where there is no trend to fit). The
verdict flip is diagnostic of window choice.

[ASSUMPTION] The legacy band remains the reference for comparability with E31/E32
and is left untouched. Any shape claim from this experiment must state its window.

---

## Anchor magnitudes

[FACT] freeze@1.0 / freeze@0.0 per seed: 9.8×, 24.5×, 27.7×, 57.8×, 75.6×
(mean 39.1×). E32 at reduced budget: 22.1×. E30 via the random_fixed proxy: 136×.
The between-seed spread is larger than the gap between these three estimates, so
the cliff should be reported as a range, never as a single number.

[FACT] `prescribed` lands at 0.00177–0.00411 and `f=0.00` (encoder frozen at
init) at 0.00097–0.00801 — the same order of magnitude. This is Ф31
(random_fixed ≈ prescribed) reproduced on measured full-budget data rather than
through the Ф12 proxy value E30 had to substitute.

---

## Setup

- **Environment:** Push-T, real `gym_pusht/PushT-v0`, obs_type='state'
- **Infrastructure:** `E32_subepoch_freeze_real/code/e32_lib.py`, imported
  unchanged (same SIGReg λ=0.09, 5-dim state encoder, AdamW lr=3e-4, wd=1e-3,
  batch 64, H=3)
- **Budget:** EP=15, NEP=200 (E32: EP=4, NEP=50)
- **Grid:** 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
  0.55, 0.60, 0.70, 0.80, 0.90, 1.00 (17 points; E32 had 10)
- **Seeds:** 42, 123, 777, 2024, 7 (same as E32)
- **Platform:** local CPU, 4 cores, torch 2.13.0+cpu, gymnasium 1.3.0,
  pymunk 6.2.1

## Files

```
code/run_seed.py                 — per-seed runner, resume-safe (skips computed points)
code/analyze_shape.py            — verbatim copy of E32's analyzer (reference verdict)
code/analyze_shape_windows.py    — same comparison across windows (this experiment's addition)
results/seed_{7,42,123,777,2024}.json — per-seed sweeps, checked in
```

## How to reproduce

```bash
for s in 42 123 777 2024 7; do
    python code/run_seed.py $s
done
python code/analyze_shape.py            # legacy band, returns STEP
python code/analyze_shape_windows.py    # window sensitivity, SLOPE on the rise
```

## Caveats

[ASSUMPTION] 5 seeds. Within the programme protocol this clears the ≥3 bar, but
the between-seed spread in both the knee position and the anchor magnitude is
large, and n=5 characterizes that spread only roughly.

[ASSUMPTION] pymunk pinned to 6.2.1 (gym_pusht 0.1.6 calls
`add_collision_handler`, removed in 6.6+). Same pin as E32, so physics is
consistent between the two, but may differ marginally from whatever produced the
original `all_results.json` behind E06/E30.

[FACT] Wall-clock varied 33–267 min per seed depending on contention with other
work on the same 4 cores (a bitbake build in a parallel WSL distro). Compute
contention affects timings only, not results.

[ASSUMPTION] The plateau criterion (first f from which the curve stays within 5%
of its max) is a post-hoc descriptive statistic chosen after seeing the curves,
not a pre-registered metric. It labels the knee; it does not test for it.

## Open

1. The JEPA initial-representation-collapse confound: T-JEPA/I-JEPA report a
   sharp early collapse-then-recover regime in the first iterations. Whether what
   is measured here inside epoch 1 is coordinate drift or that transient is not
   settled by this experiment. Prescribed and f=0.00 controls bear on it but were
   not designed for it.
2. A pre-registered knee metric, if the knee position is to be claimed rather
   than described.

## Facts

Ф46 (revised): within epoch 1 free-encoder representation damage accumulates
continuously from the first optimizer steps, is steepest over roughly the first
40% of the epoch, and then saturates. Over f∈[0.00,0.40]: 5/5 seeds strictly
monotone, linear R²=0.880, best single step 1.86× worse. The onset is not
harmless (5.8–17.1× over f∈[0.00,0.25]). Full budget EP=15/NEP=200, real
gym-pusht, 5 seeds.

## Extends / supersedes

Extends E31 (Ф46 candidate, synthetic), E32 (Ф46 on real data, reduced budget).
Supersedes both on budget and grid resolution; does not supersede their designs.
E30's 136× cliff remains the proxy-based estimate; this experiment's 9.8–75.6×
range is the measured one.
