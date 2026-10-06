# E45: E44 at 90 epochs, prescribed against free_scaled

## What this tests
Whether the provisional E44 verdict for prescribed against free_scaled changes at 90 epochs instead of 30.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф79: E45, at 90 epochs free_scaled is no worse than prescribed on the common target; registered class O4 PROVISIONAL
- STATUS (Ф83): the prescribed arm here is range-normalised, not standardised; its scale handicaps the predictor. Does not test fixation as such.
- [verified: python3 E45_long_training/code/analyze_e45.py over E45_long_training/results/cells/seed_*.json; pre-registration 4cbf9a6, pushed before the first cell] 10 seeds, prescribed and free_scaled, 90 epochs per stage, one thread per worker. Gates all true, including the first 30 stage-1 epochs equal to E44 bit for bit on every seed and arm.
- Registered statistic: GMR of final2(free_scaled) / final2(prescribed) 0.915, 95% CI [0.828, 1.010], t-test p 0.073. Class O4. Median convergence 0.916 (prescribed) and 0.940 (free_scaled), below 0.95: PROVISIONAL. On best: 0.939 [0.861, 1.025], class O4.
- [verified: same] Extra training, E44 final at 30 epochs over E45 final at 90: prescribed 1.789 [1.467, 2.183], free_scaled 2.243 [1.821, 2.763]. free_scaled has the lower final2 on 7 of 10 seeds.
- Exploratory, not registered [computed from the same seed files]: GMR of free_scaled over prescribed along E45 stage 2, 30 epochs 1.050, 60 epochs 0.914, 90 epochs 0.915.
- INTERPRETATION: the ratio moved from 1.146 (E44, 30 epochs) to 0.915 (E45, 90 epochs), toward and past 1, with the upper bound of its interval at 1.010. Neither experiment met its convergence criterion, so by the registered rules neither class is a programme fact. Within that limit the data show no prescribed advantage at dim 5 once the free encoder reads a range-normalised input, and no sign of one emerging with longer training. A criterion of 0.95 over 10 epochs may be unreachable under a constant learning rate with no schedule [INFERENCE]; a future test of this question should register the stability of the ratio across training length rather than the convergence of each loss.
- CORRECTS the Ф78 INTERPRETATION: "the remaining prescribed advantage lies between none and about 1.4x on final" holds at 30 epochs; at 90 epochs the interval is [0.828, 1.010].
- NEXT: nothing registered on this point.
- EXPLORATORY (not registered, post-hoc, recorded with Ф81): decomposition of the GMR shift between E44 and E45 on the 10 common seeds. E45 stage 1 equals E44 for its first 30 epochs and prescribed stage 2 E45[:30] equals E44 bit for bit on every seed (checked by the patch), so the free encoder at 30 epochs is shared. GMR free_scaled/prescribed on stage-2 loss: A (encoder 30, predictor 30, E44 last epoch) 1.146 [0.937, 1.402]; B (encoder 90, predictor 30, E45 stage-2 epoch 30) 1.050 [0.883, 1.249]; C (encoder 90, predictor 90) 0.915 [0.828, 1.010]. Paired ln-ratio A-B (encoder training only): mean +0.088, t 4.27 (df 9), lower in 9/10 seeds, in the direction stated before the numbers were seen (B < A). Paired B-C (predictor budget only): mean +0.138, t 1.60, lower in 7/10 seeds. A first look on seeds 1001-1007 gave A-B mean +0.082, t 3.89, 6/7. Stage-2 GMR curve in E45: ep1 0.691 [0.534, 0.893]; ep5 0.331 [0.245, 0.445]; ep10 0.682 [0.553, 0.841]; ep20 0.972 [0.836, 1.130]; ep30 1.050 [0.883, 1.249]; ep45 1.058 [0.943, 1.187]; ep60 0.914 [0.805, 1.038]; ep75 0.903 [0.811, 1.005]; ep90 0.915 [0.828, 1.010]. A second expectation stated in advance, GMR > 1 early in stage 2, is refuted: free_scaled is better over the first epochs of stage 2. That early phase is not explained.

## Setup
From EXPERIMENTS.md, section E45.

Question (Ф78). E44 classed free_scaled against prescribed as O4 but PROVISIONAL: both arms were still improving at epoch 30 (median convergence 0.863 and 0.873, below 0.95). Does the class hold when both stages are trained to convergence?

Setup. E44 unchanged except: arms prescribed and free_scaled only; 90 epochs in stage 1 and in stage 2. Library E44_common_target/code/e44_lib.py, unchanged since 483f39e; run_arm is called with epochs=90. Same seeds (42, 123, 777, 1001 to 1007), same data, split and loop. Workers run with OMP_NUM_THREADS=1, as E44 did; every seed file records the thread count.

Registered statistic and outcomes: as E44. Per seed L = ln(final2(free_scaled) / final2(prescribed)) at epoch 90; the class O1 to O5 from the 95% t interval (df 9) of GMR = exp(mean L) decides; the p value is reported only.

Convergence. conv = mean(stage-2 loss over epochs 81 to 90) / mean(epochs 71 to 80). If the median over seeds of conv for either arm is below 0.95, the class is reported as PROVISIONAL.

Validity gates, the campaign is void if any fails: prescribed stage 2 equals stage 1 bit for bit on every seed; the encoder state is unchanged by stage 2 in every cell; the first 30 epochs of stage 1 equal the E44 stage-1 history bit for bit for both arms on every seed (same code, data, seed and thread count, so any difference means the run is not what it claims to be); one pre-registration commit, ancestor of HEAD.

Named, not deciding: the registered statistic on best instead of final, with its class; per arm, E44 final at 30 epochs over E45 final at 90 epochs, as GMR with 95% interval (how much the extra training bought).

Scope: dim 5, E28 synthetic dynamics. If the class is O4 or O2 without PROVISIONAL, Ф78's reading stands as a programme fact for dim 5; if O1 or O3, Ф78 is corrected.

Code: E45_long_training/code/run_e45.py, analyze_e45.py. Results: E45_long_training/results/cells/seed_<s>.json, analysis.json.

Cost: E44 took 4 h 21 min wall for 720 epochs on the three-seed workers. E45 has 2 arms x 2 stages x 90 = 360 epochs per seed, 1080 on a three-seed worker: about 6.5 h wall on four workers.

Result (Ф79): O4 PROVISIONAL. GMR 0.915 [0.828, 1.010]; median convergence 0.916 and 0.940, below 0.95.

## Files
- `code/__pycache__`
- `code/analyze_e45.py`
- `code/patch_prereg_e45.py`
- `code/run_e45.py`
- `results/analysis.json`
- `results/cells`
- `results/worker_1.exit`
- `results/worker_1.log`
- `results/worker_2.exit`
- `results/worker_2.log`
- `results/worker_3.exit`
- `results/worker_3.log`
- `results/worker_4.exit`
- `results/worker_4.log`

## Facts
Ф79

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E45_long_training/code/run_e45.py --prereg 4cbf9a63b7cc2754847604c9f9b9f4c4019664ab --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E45_long_training/code/analyze_e45.py
```

## Status
Summary table: O4 PROVISIONAL.

- Ф79: STATUS (Ф83): the prescribed arm here is range-normalised, not standardised; its scale handicaps the predictor. Does not test fixation as such.

<!-- marker: repo-state v3 applied -->
