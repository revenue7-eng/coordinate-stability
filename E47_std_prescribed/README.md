# E47: Standardised prescribed against free_scaled on the common target at 25, 50 and 200 episodes (synthetic Push-T)

## What this tests
Standardised prescribed against free_scaled on the common target at 25, 50 and 200 episodes.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф84: E47, a standardised fixed basis beats a learned encoder on the common target, by more at low data
- STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

- Gates all true: one pre-registration commit; encoders unchanged in every stage 2; on seed 42 plain prescribed stage 2 at 200 episodes and 90 epochs equals E45 bit for bit; every E45 free_scaled history has 90 epochs. [verified: E47_std_prescribed/results/analysis.json]
- GMR free_scaled / prescribed_std on final2, n = 10, 90 epochs: 200 episodes 1.251 [1.129, 1.388], class O2; 50 episodes 1.306 [1.150, 1.484], class O2; 25 episodes 1.681 [1.452, 1.947], class O2 PROVISIONAL. Median convergence ratios: 200 std 1.028, free 0.998; 25 std 0.947, free 0.956. [verified: analysis.json]
- Г-g as registered (class at 200 is O1, O2 or O3): supported. Г-a' as registered: paired ln-ratio(25) minus ln-ratio(200) +0.295 [+0.124, +0.467], larger at 25: supported. [verified: analysis.json]
- Ratio stability, named: GMR at epoch 60 against 90, 200 episodes 1.169 against 1.251; 25 episodes 1.717 against 1.681. [verified: analysis.json]
- Known asymmetry (E44 block): the common target is the prescribed features of s(t+3), an affine function of the prescribed_std latent, so its predictor receives the target coordinates in its input while free_scaled must also decode them. The comparison is an upper bound in favour of prescribed. Axis alignment alone does not carry it (Ф78, rotated arm 1.00x), linear availability of the target is not excluded.
- INTERPRETATION: with both inputs well scaled, fixing the full state as the latent beats a learned encoder on the same input by about 1.25x at 200 episodes and 1.7x at 25, when the target is linear in the fixed coordinates. The old claim of sample efficiency holds in direction, at tens of percent rather than orders of magnitude. Synthetic dynamics, dim 5, fully observed state.
- NEXT: E48, the same comparison on a target nonlinear in the prescribed coordinates.

## Setup
From EXPERIMENTS.md, section E47.

Question. Ф83: with the prescribed features standardised, is there a prescribed advantage on the common target, and is it larger at low data?

Arms. prescribed_std: the five prescribed features standardised with mean and std of the training split of that seed and size; encoder fixed, stage 2 only (as E46 part B). free_scaled: both stages as E45, 90 epochs each, at 25 and 50 episodes; at 200 episodes taken from E45 cells (same seeds, data and code path). Common target and loop as E44. 90 epochs per stage. Seeds 42, 123, 777, 1001 to 1007 (n = 10). OMP_NUM_THREADS=1 per worker. Fewer episodes at fixed epochs also means fewer optimiser steps; not separated.

Code: E47_std_prescribed/code/run_e47.py (imports e44_lib.py unchanged), analyze_e47.py.

Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed_std), GMR with 95% t interval (df 9), classes O1 to O5 as E44. Convergence: conv = mean(stage-2 loss, epochs 86 to 90) / mean(epochs 81 to 85); PROVISIONAL if the median over seeds for either arm is below 0.95. Г-g (a standardised fixed basis is better on the common target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5. Г-a' (the advantage is larger at low data): paired difference ln-ratio(25) minus ln-ratio(200) over seeds with 95% t interval; "larger at 25" if the lower bound is above 0, "smaller at 25" if the upper bound is below 0, otherwise not resolved.

Named, not deciding: GMR at epoch 60 against epoch 90 per size (ratio stability, Ф79 INTERPRETATION); the 50-episode class.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; on seed 42 plain prescribed stage 2 at 200 episodes and 90 epochs equals E45 prescribed s2_hist bit for bit; every E45 free_scaled history has 90 epochs.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. 200 episodes O2, 50 O2, 25 O2 PROVISIONAL; low-data difference larger at 25. Г-g and Г-a' supported as registered. See Ф84.

## Files
- `code/__pycache__`
- `code/analyze_e47.py`
- `code/patch_e46res_e47prereg.py`
- `code/run_e47.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.log`
- `results/worker_2.log`
- `results/worker_3.log`
- `results/worker_4.log`

## Facts
Ф84

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E47_std_prescribed/code/run_e47.py --prereg 83a16db15adf7e3871791954b5d69c633d4bb7cc --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E47_std_prescribed/code/analyze_e47.py
```

## Status
Summary table: state vs frozen JEPA latent (Н5).

- Ф84: STATUS (Н5): prescribed_std is the standardised state with no encoder. The result reads: a frozen JEPA latent of the full state is complete but laid out worse than the state itself. It does not test fixed coordinates of a learned representation.

<!-- marker: repo-state v3 applied -->
