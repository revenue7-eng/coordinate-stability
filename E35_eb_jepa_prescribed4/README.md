# Experiment 35: EB-JEPA Two Rooms — prescribed_4 (coordinate completeness)

## Status
ON HOLD (24-25.08.2026). The premise this experiment was designed against does not hold: re-evaluated with a shared eval path, prescribed_2 and free both give 0.55 planning SR on the same twenty episodes (Ф57). The 0% for prescribed_2 was an evaluation artefact. Planning SR at n=20 also fails to separate the free encoder, which holds wall_x at R2 0.969 (Ф56), from prescribed_2, which holds nothing about the obstacle - so the metric has no demonstrated power to detect what prescribed_4 would add. Resuming requires a metric with established sensitivity and a quantitative falsifier; neither exists yet. Implementation exists (`code/run_experiment_v4_windows.py`); no condition has been trained to completion.

## What this tests
As designed: E34 appeared to show prescribed_2 = (x_a, y_a) reaching 0% planning SR against 55% for the free pixel encoder, and this experiment adds the missing state, prescribed_4 = (x_a, y_a, wall_x, door_y).

That contrast has since been withdrawn (Ф57): both branches reach 0.55. What prescribed_4 would add over prescribed_2 is now an open question with no baseline gap to close, and no measurement showing that the planned metric would register it.

## Hypothesis Г25
Prescribed advantage requires coordinate completeness with respect to the downstream task. Axis identifiability does not substitute for completeness of the coordinate description.

Source of record: `EVIDENCE.md`, Г25 (status: OPEN, untested; support in this environment withdrawn, assigned test on hold). This is a translation of the registry entry; the registry is authoritative.

## Falsifier
Defined 07.09.2026 from a power calculation, superseding NOT DEFINED. The original criterion - prescribed_4 yielding SR near 0% - was written against a baseline of 0% that turned out to be an evaluation artefact (Ф57). Against a baseline of 0.55 the criterion has to be quantitative, and writing one requires knowing what difference the metric can resolve at n=20. That is the blocking question, not the encoder width.

Resolution of the metric at n=20 (exact McNemar, two-sided, alpha=0.05, paired on
`setup_seed(1)`, baseline SR 0.55). Power against a true effect of:
+0.15 -> 0.067; +0.20 -> 0.196; +0.25 -> 0.383; +0.30 -> 0.584; +0.35 -> 0.755,
all assuming zero regressions. Allowing one regression in twenty (p10=0.05), even
a true SR of 1.00 gives 0.785. Twenty episodes therefore cannot support any
falsifier: below +0.35 the design returns a non-significant result whether or not
the effect exists.

n required for power 0.80, p10=0.05: +0.10 -> 168; +0.15 -> 92; +0.20 -> 61;
+0.25 -> 45; +0.30 -> 36.

Falsifier: n=60 per condition, paired, exact McNemar, two-sided alpha=0.05. G25 is
unsupported in this environment if prescribed_4 does not exceed free by at least
+0.20 SR with a significant test. Effects below +0.20 are outside the resolution of
this design; that limit is declared before the run, not read off the result.

The continuous alternative does not help: on the re-evaluated runs distances are
bimodal (success <= 1.108, failure >= 5.411, nothing between), so mean_dist carries
the same information as binary SR and inherits the same n requirement.

## Conditions implemented in v4
`free`, `prescribed` (= prescribed_2), `prescribed_4`, `hybrid`, `hybrid_4`, plus ablations `prescribed_no_idm`, `prescribed_no_vicreg`, `prescribed_no_sim`, `prescribed_4_no_sim`.

`wall_x` and `door_y` are constant along T within a trajectory but vary across samples (`fix_wall=False` in `data_config`).

## Required before launch

### 1. Normalization — resolved, not blocking
v4 applies z-score to all four channels, reusing EB-JEPA's agent-coordinate constants
(mean [31.5863, 32.0618], std [16.1025, 16.1353]). Agent `loc` already arrives normalized
from `dataset.normalizer.normalize_location` when `normalize=True`, verified by measurement
(see "Channel scaling" below).

The earlier requirement to run min-max as a parallel second condition is withdrawn. It
rested on Ф40 (standardization 15x worse) and on Г14(b) (min-max mandatory). Ф40 did not
reproduce under direct re-runs on 23.08.2026: the registry comparison mixes two conditions
that differ in target normalization (a 111.6x target-variance factor) and in epoch count
(30 vs 20). Re-measured at equal epochs, the largest reproducible harm from standardization
is 1.23x, and only in one of three schemes. The `EVIDENCE.md` entry for Ф40 still carries
its original wording as of this writing and is pending amendment to NOT REPRODUCIBLE;
Г14(b) is not quantitatively supported for this claim. With the basis gone,
the parallel condition is not justified and the run budget is not doubled.

The residual ambiguity stated below under "Basis of the normalization requirement" also
falls away with it: an SR ~ 0% outcome under z-score is no longer confounded by a registry
entry that does not reproduce.

### 2. Secondary success criterion — done
Binary SR is too coarse for the falsifier. The per-episode distances quoted here from the
first E34 evaluation (8.36 / 9.12 / 13.30 against a free mean of 9.78) are superseded: that
evaluation is the one Ф57 replaces, and its per-episode figures cannot be traced to a stored
artefact. On the re-evaluated runs the distances are bimodal - every success ends at 1.108
or below, every failure at 5.411 or above - so on this data binary SR and the distance array
carry nearly the same information, and the coarseness argument needs remaking on whatever
metric replaces SR.

The full `distances` array was already written by `planning_eval_v4.py`. Added 24.08.2026:
per-episode geometry (`wall_x`, `hole_y`, start, goal and final position, all in raw pixel
coordinates). Geometry is not recoverable after a run, and without it a final distance
cannot be normalized against the scale of the episode. With both arrays stored, any summary
— median, quartiles, below-threshold fractions at any threshold — is computable afterwards
without a re-run.

Deliberately not done here: fixing a set of thresholds. Choosing them now without a decision
rule would only move the post-hoc choice earlier. The threshold for a non-trivial outcome,
and the rule that reads it, belong in the spec before launch. `success_rate` continues to
use the upstream threshold of 4.5 px (`two_rooms/env.py:106`), unchanged, so comparability
with the 55% free figure from E34 is preserved.

### 3. Capacity — decided (24.08.2026)
prescribed_2 used 199,168 encoder parameters against 1,426,096 for free (7.2x); prescribed_4
has 199,680, the 512 difference being two extra input channels into the first Linear. The MLP
width stays as inherited from v3. Rationale for the record: matching E34 is deliberate, so
that one variable changes relative to that experiment; parity with free is undefined, since
no correspondence exists between a pixel encoder and an MLP over four inputs; and if both
conditions come out at zero, interpretation is limited by the joint ceiling of the two
regardless of width.

No latent probe is added in E35. In the prescribed branch the probe recovers the encoder's
own input — the same defect for which `probe_loss` is excluded below.

## Metric that does not apply
`probe_loss` is not interpretable in any prescribed branch: the probe (MLPXYHead on the detached latent, MSE against agent loc) recovers the encoder's own input. Compare within the free condition only, or change the probe target to something outside the prescribed axes.

## Follow-up (conditional on outcome)
Withdrawn. All three branches were cut against an SR baseline of 0%, which does not exist
(Ф57). Any follow-up structure has to be rebuilt on the metric that replaces SR.

## Files
- `code/run_experiment_v4_windows.py` — extends v3 with prescribed_4 / hybrid_4; training only
- `code/planning_eval_v4.py` — planning evaluation, run separately after training

## Upstream and reproducibility

Built on `facebookresearch/eb_jepa` @ `966e61e9285b3a876f49b9774e9720d9a99a7925`
(v0.1.1, Apache 2.0, arXiv:2602.03604). The local working copy is byte-identical
to that commit across all code, configs and tests, with one exception:
`pyproject.toml` relaxes `torch==2.6.0` to `torch` and `requires-python == 3.12.*`
to `>= 3.12`. Consequence: the eval environment runs torch 2.13.0+cpu, not the
pinned 2.6.0. The torch version used for the August training runs is not recorded.

Environment configs (`train.yaml`, `eval.yaml`,
`eb_jepa/datasets/two_rooms/data_config.yaml`) are upstream and not vendored here;
obtain them from the commit above. Only `cfgs/planning_mppi.yaml` is vendored, and
it differs from upstream by exactly one line: `sum_all_diffs: true` -> `false`.

Checkpoints (epoch 11, not in version control):
- prescribed: md5 `37aa06ee71f4a6e5754d810bff6469ea`
- free: md5 `7c6a27ad873ca5c57c0d717459601b38`

## The training script does not evaluate planning (verified 24.08.2026)
`run_experiment_v4_windows.py` imports `main_eval` but never calls it, and defines
`set_locations_for_planning` without calling it; `cfg.meta.enable_plan_eval` is set to False
and the per-epoch `sr`/`md` are hardcoded to -1.0. The same holds for v3. A completed
training run therefore yields checkpoints and `pred`/`reg` curves, but no SR and no
distances. Planning evaluation is `code/planning_eval_v4.py`, run afterwards against a
saved checkpoint. In E34 this step was a notebook (`code/eb_jepa_planning_eval.ipynb`).

Changes to `planning_eval_v4.py` on 24.08.2026:
- Agent and goal locations are now passed through `normalizer.normalize_location` before
  reaching the encoder. Previously they were taken raw from `env.info` while the model had
  been trained on normalized ones. The call must run on shape [2]: `normalize_location`
  broadcasts over the last axis, so a [1, 2, 1] input silently becomes [1, 2, 2] with no
  error raised.
- Four-channel support: `wall_x` and `hole_y` are read from the env (they are set in
  `reset()` and constant within an episode; note the attribute is `hole_y`, not `door_y`)
  and passed through the training script's own `build_loc_input`, so the z-score constants
  are the same object rather than a copy.
- The duplicated `PrescribedEncoder`, `HybridEncoder`, `CONDITIONS` and `build_encoder`
  are removed in favour of importing them from `run_experiment_v4_windows`. All nine
  conditions are now available to the evaluator, and `prescribed_dim` is threaded through
  instead of being hardcoded to 2. `PrescribedJEPA` stays local: the planning variant needs
  an `expand()` over the MPPI candidate batch that the training variant does not.
- Per-episode geometry is recorded (see section 2).

## Related
- E34 — prescribed_2 vs free. The observation this was designed to respond to (Н1) is refuted;
  the re-evaluation that replaced it is Ф57, artefacts under
  `E34_eb_jepa_planning/results/{free,prescribed}_reeval_20260824/`

## Basis of the normalization requirement — superseded (24.08.2026)
Section 1 originally rested on Ф40 and Г14. A scope check on 22.08.2026 already found the
extrapolation weak: Ф40 is a single measurement on synthetic data from Paper 1 /
random_axes_control, 200 episodes, regression loss; Г14 is CONFIRMED but based on Push-T and
the double pendulum, dim 1-11, again on regression-loss ratios. Neither covers EB-JEPA,
Two Rooms, or planning SR.

The re-runs of 23.08.2026 went further: Ф40 does not reproduce at all. The requirement it
supported is withdrawn rather than merely qualified. This section is kept as a record of how
the requirement was arrived at and removed; it imposes nothing on the run.

## Relation to Г14
Г25 (coordinate completeness) restates clause (c) of Г14 — that the coordinates must carry task-relevant information — and tests it on a third environment. The two entries are not independent; Г25 is a subset of Г14(c) evaluated outside the domain where Г14 was confirmed.

## Channel scaling of wall_x / door_y — measured (24.08.2026)

`build_loc_input` normalizes channels 3-4 with the agent-coordinate constants
LOC_MEAN_X=31.5863 / LOC_STD_X=16.1025 and LOC_MEAN_Y=32.0618 / LOC_STD_Y=16.1353.
These are the same values held by `Normalizer` (`two_rooms/normalizer.py:11-12`), which is
what applies the z-score to the agent coordinates. Each channel takes the statistics of its
own axis (wall_x with X, door_y with Y). The two paths are not bit-identical: `Normalizer`
divides by `std + 1e-6`, `build_loc_input` by the bare constant. The difference appears in
the sixth decimal and is why `planning_eval_v4.py` calls each path for the channels it owns
rather than routing all four through one of them.

Measured over 20 training batches (1280 samples, batch_size 64, `fix_wall: false`,
`normalize: true`):

- wall_x: 25 distinct values, [20, 44], mean 32.2492, std 7.2899 -> after z-score std 0.4527
- door_y: 45 distinct values, [10, 54], mean 31.7727, std 12.7838 -> after z-score std 0.7923
- agent loc as fed to the encoder: mean (0.0082, 0.0521), std (0.9955, 1.0488), range within +/-1.77

The agent coordinates arrive already z-scored by `dataset.normalizer`, confirming the
assumption stated in the `build_loc_input` docstring. Channels 3-4 therefore enter at
0.45x and 0.79x the dynamic range of channels 1-2. The imbalance across all four channels
spans a factor of 2.2.

An earlier version of this section called the reuse of agent statistics a defect and
proposed per-channel statistics as a four-line repair. That is withdrawn. A per-channel
affine change of the input is absorbed by the first `nn.Linear`, which sees the raw four
vector with no preceding normalization: the class of representable functions is identical
under either scheme, so no geometric relation is lost or gained. What differs is only the
prior — the weight ratio that expresses a given function — which is an argument about
inductive bias, not about correctness, and no measurement supports it either way.
The measured 2.2x spread is not a pathology in either direction.

The current scheme also has one property the alternative lacks: under a shared transform,
the difference between the agent's x and the wall's x stays proportional to the pixel
distance between them. Per-channel statistics would rescale the two independently. This is
an observation about representation, not a demonstration that either scheme trains better.

Decision: channels 3-4 keep the agent-coordinate constants. The scheme is a recorded design
choice (rationale in `build_loc_input`, referencing work_plan_2026_04_30.md); changing it
mid-experiment would require its own justification, which does not exist.

This section reports measurements of the dataset as configured; it is not an experimental
result and carries no Ф number. It is invalidated if img_size, wall_padding, or
door_padding change.

Not applicable to B1: that control ran on the prescribed_dim=2 checkpoint
(experiment_mode.txt = "prescribed"; encoder.projection.0.weight has shape (256, 2)),
which never received wall_x or door_y. Ф56 is unaffected by this finding.
