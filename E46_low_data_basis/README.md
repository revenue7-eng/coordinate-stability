# E46: Low data and the basis of the prescribed latent (synthetic Push-T, E44 protocol)

## What this tests
Whether prescribed wins at low data (25 and 50 episodes, Г-a) and whether the basis of the prescribed latent (range-normalised, sin and cos, standardised) changes the result at 200 episodes (Г-b).

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф83: E46, plain prescribed loses at low data; standardising the prescribed features removes the early free advantage and puts prescribed ahead at 200 episodes

- Gates all true: one pre-registration commit, prescribed stage 2 equals stage 1 at 25 and 50 episodes, encoders unchanged in every stage 2, prescribed_gate equals E44 prescribed stage 2 bit for bit on every seed. [verified: E46_low_data_basis/results/analysis.json]
- Г-a as registered, GMR free_scaled / plain prescribed on final2, n = 10: 25 episodes 0.375 [0.283, 0.495], class O5 PROVISIONAL; 50 episodes 0.865 [0.613, 1.222], class O4 PROVISIONAL; 200 episodes (E44) 1.146 [0.937, 1.402]. By the registered rule (O4 or O5 at 25) Г-a is refuted for range-normalised prescribed. [verified: analysis.json]
- Г-b, early GMR free_scaled / arm over stage-2 epochs 1 to 10 at 200 episodes: plain prescribed (E44) 0.427 [0.367, 0.498]; prescribed_sincos 0.262 [0.238, 0.289], early free advantage present; prescribed_std 1.233 [1.154, 1.316], early free advantage removed. [verified: analysis.json]
- Final at 200 episodes, reported and not deciding (no convergence check registered for part B): prescribed_std 1.218 [1.096, 1.352], class O2; prescribed_sincos 0.512 [0.441, 0.594], class O5. [verified: analysis.json]
- INTERPRETATION: the early free advantage recorded as EXPLORATORY under Ф79 is the scale of the range-normalised prescribed features (values in [0, 1], small variance), not the wrap of the angle. This is the mirror of Ф76 on the prescribed side [INFERENCE, from the std arm; mechanism not isolated further]. E44, E45 and the Г-a verdict above compare a poorly scaled prescribed latent with a well scaled free one; their classes do not test fixation as such.
- NEXT: E47, standardised prescribed against free_scaled at 25, 50 and 200 episodes, 90 epochs.

## Setup
From EXPERIMENTS.md, section E46.

Questions. Г-a: is a prescribed advantage on the common target visible at low data (25, 50 episodes), where the old claim of sample efficiency lived (Ф38 withdrawn, Ф80)? Г-b: in E45 stage 2, free_scaled learns faster than prescribed over the first epochs (Ф79 EXPLORATORY). Is that the basis of the prescribed latent: the wrap of theta/2pi (arm prescribed_sincos) or the small scale of range-normalised features (arm prescribed_std)? Rotation does not change it (Ф78, 1.00x), so only non-orthogonal changes are tested.

Code: E46_low_data_basis/code/run_e46.py imports E44_common_target/code/e44_lib.py unchanged; analyze_e46.py. Seeds 42, 123, 777, 1001 to 1007 (n = 10), as E44. 30 epochs per stage, OMP_NUM_THREADS=1 per worker.

Part A (Г-a). prescribed and free_scaled, both stages exactly as E44 run_arm, at EPISODES 25 and 50 (data synth(n, seed)). Fewer episodes at fixed epochs also means fewer optimiser steps, as in the old E05a design; the two are not separated here. Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed), GMR with 95% t interval (df 9), classes O1 to O5 as E44; PROVISIONAL if the median convergence ratio (E44 definition) is below 0.95. Г-a is supported if the class at 25 episodes is O1, O2 or O3; refuted if O4 or O5 at 25.

Part B (Г-b). At 200 episodes, stage 2 only (encoder fixed, fresh predictor on the common target, same seed and loop as E44 stage 2). Arms: prescribed_gate (plain prescribed); prescribed_sincos = (x_a, y_a, x_b, y_b)/512 and sin, cos of the angle (6 dims; the predictor output layer is 5 wide); prescribed_std = the five prescribed features standardised with mean and std of the seed's training split. Reference: E44 free_scaled stage 2 of the same seed. Registered statistic per arm: early = mean over stage-2 epochs 1 to 10 of ln(free_scaled / arm), GMR over seeds with 95% t interval. "Early free advantage removed" if the upper bound is at least 1, "present" if below 1. For E44 prescribed this is computed for reference. Final-epoch GMR and class are reported, not deciding.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; in part A prescribed stage 2 equals stage 1 bit for bit; the encoder is tensor-equal before and after every stage 2; prescribed_gate stage 2 equals E44 prescribed stage 2 bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. Г-a: 25 episodes O5 PROVISIONAL, 50 episodes O4 PROVISIONAL (plain prescribed); refuted as registered. Г-b: prescribed_std early free advantage removed, prescribed_sincos early free advantage present. See Ф83.

## Files
- `code/__pycache__`
- `code/analyze_e46.py`
- `code/patch_prereg_e46.py`
- `code/run_e46.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.log`
- `results/worker_2.log`
- `results/worker_3.log`
- `results/worker_4.log`

## Facts
Ф83

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E46_low_data_basis/code/run_e46.py --prereg e1bdc6f20d6cae28592fd2f303a8d0e39db7f2d6 --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E46_low_data_basis/code/analyze_e46.py
```

## Status
Summary table: scale of the fixed latent matters.

- Ф83: no correction recorded in the registry; stands as recorded

<!-- marker: repo-state v3 applied -->
