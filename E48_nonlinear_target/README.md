# E48: E47 on a target nonlinear in the prescribed coordinates (synthetic Push-T)

## What this tests
Whether the E47 advantage of a standardised fixed basis survives a target nonlinear in the prescribed coordinates.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф85: E48, the advantage of a standardised fixed basis survives a target nonlinear in the fixed coordinates
- STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

- Gates all true: one pre-registration commit; encoders unchanged in every stage 2; free_scaled stage 1 at 200 episodes equals E45 bit for bit on every seed. [verified: E48_nonlinear_target/results/analysis.json]
- Target g(s_t+3) = (agent-block distance, sin theta, cos theta, agent position in the block frame). GMR free_scaled / prescribed_std on final2, n = 10, 90 epochs: 200 episodes 1.436 [1.256, 1.642], class O2; 25 episodes 1.681 [1.358, 2.082], class O2 PROVISIONAL. [verified: analysis.json]
- Г-i as registered (class at 200 is O1, O2 or O3): supported. Paired ln-ratio(25) minus ln-ratio(200): +0.158 [-0.097, +0.413], not resolved. [verified: analysis.json]
- Named: GMR on the nonlinear target divided by the E47 GMR, 200 episodes 1.147, 25 episodes 1.000. [verified: analysis.json]
- INTERPRETATION: the linear availability of the target named in Ф84 does not carry the advantage; the advantage is as large or larger when the predictor must compute a nonlinear function of the fixed coordinates. The low-data increase of Ф84 is not reproduced as resolved on this target. Synthetic dynamics, dim 5, fully observed state.
- NEXT: E49, whether the frozen JEPA latent of free_scaled loses state information (completeness) and whether the same encoder trained end to end on the target closes the gap.

## Setup
From EXPERIMENTS.md, section E48.

Question. Ф84 names the linear availability of the common target to prescribed_std as not excluded. Does the advantage of a standardised fixed basis survive when the target is nonlinear in the prescribed coordinates?

Target. g(s_{t+3}) = (d, sin theta, cos theta, u, v): d the agent-block distance, (u, v) the agent position relative to the block in the block's frame, positions divided by 512. Every component is nonlinear in the prescribed features. Stage 1 (own latent) is unchanged.

Arms and protocol as E47: prescribed_std (stage 2 only), free_scaled (both stages), 90 epochs per stage, at 25 and 200 episodes; free_scaled at 200 is rerun because its stage 2 target changes. Seeds 42, 123, 777, 1001 to 1007 (n = 10). Code: E48_nonlinear_target/code/run_e48.py (imports e44_lib.py unchanged; the world model's external target is replaced in a subclass), analyze_e48.py.

Registered statistic, classes, convergence rule and paired low-data difference exactly as E47. Г-i (the advantage survives a nonlinear target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5.

Named, not deciding: GMR on the nonlinear target divided by the E47 GMR at the same size (the share of the E47 advantage that the linear target carried); GMR at epoch 60.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; free_scaled stage 1 at 200 episodes equals E45 free_scaled s1_hist bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. 200 episodes O2, 25 O2 PROVISIONAL; low-data difference not resolved. Г-i supported as registered. See Ф85.

## Files
- `code/__pycache__`
- `code/analyze_e48.py`
- `code/patch_e47res_e48prereg.py`
- `code/run_e48.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.log`
- `results/worker_2.log`
- `results/worker_3.log`
- `results/worker_4.log`

## Facts
Ф85

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E48_nonlinear_target/code/run_e48.py --prereg 9dd2a819b563a3a636e3fb64450a8e864548dc43 --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E48_nonlinear_target/code/analyze_e48.py
```

## Status
Summary table: state vs frozen JEPA latent (Н5).

- Ф85: STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

<!-- marker: repo-state v3 applied -->
