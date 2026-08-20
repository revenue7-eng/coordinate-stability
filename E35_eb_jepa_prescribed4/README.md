# Experiment 35: EB-JEPA Two Rooms — prescribed_4 (coordinate completeness)

## Status
READY_TO_START. Implementation exists (`code/run_experiment_v4_windows.py`, written 30 April 2026) and has never been run.

## What this tests
E34 showed prescribed_2 = (x_a, y_a) reaching 0% planning SR against 55% for the free pixel encoder. The prescribed latent contained no obstacle information. This experiment adds the missing state: prescribed_4 = (x_a, y_a, wall_x, door_y).

## Hypothesis Г25
Prescribed advantage requires coordinate completeness with respect to the downstream task. Axis identifiability does not substitute for completeness of the coordinate description.

Source of record: `EVIDENCE.md`, Г25 (status: OPEN, assigned to E35). This is a translation of the registry entry; the registry is authoritative.

## Falsifier
If prescribed_4 = (x_a, y_a, wall_x, door_y) yields SR ≈ 0% on Two Rooms, the completeness hypothesis is refuted and the problem lies deeper: architecture mismatch, insufficient capacity, or a fundamental limitation of the prescribed approach in environments with obstacles.

## Conditions implemented in v4
`free`, `prescribed` (= prescribed_2), `prescribed_4`, `hybrid`, `hybrid_4`, plus ablations `prescribed_no_idm`, `prescribed_no_vicreg`, `prescribed_no_sim`, `prescribed_4_no_sim`.

`wall_x` and `door_y` are constant along T within a trajectory but vary across samples (`fix_wall=False` in `data_config`).

## Required before launch

### 1. Normalization — blocking
v4 hardcodes z-score for all four channels, reusing EB-JEPA's agent-coordinate constants (mean ~ [31.59, 32.06], std ~ [16.10, 16.14]). Agent `loc` already arrives normalized from `dataset.normalizer.normalize_location` when `normalize=True`.

This conflicts with two registry entries: Ф40 (standardization is 15x worse) and Г14 (min-max [0,1] is a necessary condition for prescribed advantage). Under z-score alone, an SR ~ 0% outcome is unreadable: insufficient coordinates and unfavourable normalization become indistinguishable.

Amendment: run min-max as a parallel second condition, not as a follow-up conditional on failure. This is not a config switch. It requires `normalize=False` in `data_config`, a min-max path in `build_loc_input` covering all four channels, and a check that the normalizer path used by `planning.py` stays consistent with it. Pre-training verification: print per-channel ranges — min-max gives exactly [0, 1].

Note: channels 3-4 (`wall_x`, `door_y`) are currently normalized with agent-coordinate statistics rather than their own.

### 2. Secondary success criterion
Binary SR is too coarse for the falsifier. In E34 the prescribed condition produced per-episode distances of 8.36 / 9.12 / 13.30 against a free mean of 9.78 — near-misses recorded identically to trajectories ending at 71.37. A prescribed_4 run that halves the distance distribution without crossing the success threshold would read as refutation while being partial confirmation.

Amendment: record the median and the below-threshold fraction of the `distances` array alongside SR. The array is already written; this costs nothing.

### 3. Capacity — decide and record
prescribed_2 used 199,168 encoder parameters against 1,426,096 for free (7.2x). At SR ~ 0% the capacity confound is indistinguishable from informational incompleteness. Fix the MLP width deliberately and record the choice in the spec rather than inheriting it from v3.

## Metric that does not apply
`probe_loss` is not interpretable in any prescribed branch: the probe (MLPXYHead on the detached latent, MSE against agent loc) recovers the encoder's own input. Compare within the free condition only, or change the probe target to something outside the prescribed axes.

## Follow-up (conditional on outcome)
- SR > 30%: prescribed_3 = (x_a, y_a, wall_x) — separate the wall from the door
- SR ~ 0%: hybrid run (hybrid_4 is implemented)
- SR 5-30%: additional seed for variance estimation

## Cost
60-100 h CPU, background, auto-resume (dual local + Drive checkpointing). Doubled by the parallel min-max condition.

## Files
- `code/run_experiment_v4_windows.py` — extends v3 with prescribed_4 / hybrid_4; not yet run

## Related
- E34 — prescribed_2 vs free, the observation this responds to (Н1)
