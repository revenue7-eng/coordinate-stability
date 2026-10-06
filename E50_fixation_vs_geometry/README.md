# E50: Fixation or geometry: a fixed, nonlinearly warped, complete encoder on the E48 target (synthetic Push-T)

## What this tests
Whether the advantage of prescribed_std comes from fixed coordinates (Г-k) or from the simple layout of the state in them (Г-l), using a fixed but nonlinearly warped complete encoder and a fairly trained end-to-end encoder.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Н5: E50, fixation or geometry: VOID (completeness gate failed); exploratory observations
- Gates: one pre-registration commit, true; fixed_warped encoder tensor-equal before and after training, true; fixed_warped complete (minimum median MLP probe R2 at least 0.99), false: minimum 0.983. The campaign is void as registered. [verified: python3 E50_fixation_vs_geometry/code/analyze_e50.py; E50_fixation_vs_geometry/results/analysis.json]
- Instrument defect: the 0.99 gate was set without checking that the probe can invert a warp of this strength; the analysis printed a verdict line although the gates failed. The gate is not rewritten after the data.
- Probes of fixed_warped, median over 10 seeds. MLP R2: x_a 0.989, y_a 0.989, x_b 0.989, y_b 0.989, sin 0.983, cos 0.985. Linear R2: x_a 0.916, y_a 0.927, x_b 0.920, y_b 0.917, sin 0.606, cos 0.003.
- Exploratory, not a verdict: fixed_warped / prescribed_std 7.507 [6.613, 8.522], O2; frozen JEPA (E48) / fixed_warped 0.191 [0.163, 0.224], O5; free_e2e_fair / prescribed_std 0.736 [0.639, 0.848], O5; free_e2e_fair / frozen JEPA 0.512 [0.449, 0.585], O5.
- Reading: the warped fixed basis is far worse than prescribed_std and worse than the learned latent, which points to layout (Г-l) rather than fixation (Г-k); its warp is stronger than that of the JEPA latent (linear R2 of positions about 0.92 against about 0.98), so this is not a matched test. A learned encoder trained end to end on the target with an equal budget and no SIGReg is ahead of prescribed_std, so the E49 limit (1) was under-training, as named in Ф86.
- Scope of the E46 to E50 series: on fully observed synthetic Push-T prescribed_std is the state itself, standardised, with no encoder. The series measures a frozen JEPA latent of the state against the state, not fixed coordinates of a learned representation. Hypotheses Г-a to Г-l were built adaptively, each after reading the previous campaign, on the same 10 seeds and one generator, without a multiplicity correction.

## Setup
From EXPERIMENTS.md, section E50.

Question. Ф86: is the advantage of prescribed_std that its coordinates are fixed (Г-k) or that the state is laid out simply in them (Г-l)?

Arms, 200 episodes, E48 target, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. fixed_warped: frozen encoder f -> f Q1 -> h + A tanh(B h) -> Q2 -> h + A tanh(B h) -> Q3, then standardised on the training split; f the standardised prescribed features; A = 1.5, B = 2 (each step strictly monotone, so the map is invertible); Q1, Q2, Q3 random orthogonal from a private generator seeded 50000 + seed. Fixed like prescribed_std, complete, nonlinearly laid out like the learned latent. Stage 2 only, 90 epochs, E44 loop with SIGReg weight 0.09. Probed as in E49.
2. free_e2e_fair: FreeEncoderScaled trained end to end on the target for 180 epochs (the total budget of frozen JEPA), SIGReg weight 0, otherwise the E44 loop.
prescribed_std and frozen JEPA free_scaled are taken from E48 cells (same seeds, data, target).
Code: E50_fixation_vs_geometry/code/run_e50.py (imports e44_lib.py unchanged), analyze_e50.py.

Registered statistics: GMR fixed_warped / prescribed_std and GMR frozen JEPA (E48) / fixed_warped on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: fixation (Г-k) if fixed_warped / prescribed_std is O4 and frozen JEPA / fixed_warped is O1, O2 or O3; geometry (Г-l) if fixed_warped / prescribed_std is O1, O2 or O3 and frozen JEPA / fixed_warped is O4 or O5; otherwise mixed. The verdict is not deciding if the warp is too weak: minimum over the four positions of the median linear probe R2 of fixed_warped is at least 0.99.
Named, not deciding: GMR free_e2e_fair / prescribed_std and / frozen JEPA; linear and MLP probes of fixed_warped; median convergence ratios.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the fixed_warped encoder is tensor-equal before and after training; fixed_warped is complete (minimum median MLP probe R2 at least 0.99).

Cost: not estimated in advance; the monitor reports the measured rate.

Result: VOID, completeness gate failed (minimum median MLP probe R2 of fixed_warped 0.983 against 0.99). Exploratory numbers and the reading are in EVIDENCE Н5. Data: E50_fixation_vs_geometry/results/cells, analysis.json.

## Files
- `code/__pycache__`
- `code/analyze_e50.py`
- `code/patch_e49res_e50prereg.py`
- `code/run_e50.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.log`
- `results/worker_2.log`
- `results/worker_3.log`
- `results/worker_4.log`

## Facts
Н5

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E50_fixation_vs_geometry/code/run_e50.py --prereg a5c7552c1a9320e62906f04a0b18fbf48e31bb63 --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E50_fixation_vs_geometry/code/analyze_e50.py
```

## Status
Summary table: VOID.

- Н5: no correction recorded in the registry; stands as recorded

<!-- marker: repo-state v3 applied -->
