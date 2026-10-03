#!/usr/bin/env python3
# marker: E46 v1
import re
from pathlib import Path

BLOCK = """### E46. Low data and the basis of the prescribed latent (synthetic Push-T, E44 protocol): PRE-REGISTERED

Questions. Г-a: is a prescribed advantage on the common target visible at low data (25, 50 episodes), where the old claim of sample efficiency lived (Ф38 withdrawn, Ф80)? Г-b: in E45 stage 2, free_scaled learns faster than prescribed over the first epochs (Ф79 EXPLORATORY). Is that the basis of the prescribed latent: the wrap of theta/2pi (arm prescribed_sincos) or the small scale of range-normalised features (arm prescribed_std)? Rotation does not change it (Ф78, 1.00x), so only non-orthogonal changes are tested.

Code: E46_low_data_basis/code/run_e46.py imports E44_common_target/code/e44_lib.py unchanged; analyze_e46.py. Seeds 42, 123, 777, 1001 to 1007 (n = 10), as E44. 30 epochs per stage, OMP_NUM_THREADS=1 per worker.

Part A (Г-a). prescribed and free_scaled, both stages exactly as E44 run_arm, at EPISODES 25 and 50 (data synth(n, seed)). Fewer episodes at fixed epochs also means fewer optimiser steps, as in the old E05a design; the two are not separated here. Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed), GMR with 95% t interval (df 9), classes O1 to O5 as E44; PROVISIONAL if the median convergence ratio (E44 definition) is below 0.95. Г-a is supported if the class at 25 episodes is O1, O2 or O3; refuted if O4 or O5 at 25.

Part B (Г-b). At 200 episodes, stage 2 only (encoder fixed, fresh predictor on the common target, same seed and loop as E44 stage 2). Arms: prescribed_gate (plain prescribed); prescribed_sincos = (x_a, y_a, x_b, y_b)/512 and sin, cos of the angle (6 dims; the predictor output layer is 5 wide); prescribed_std = the five prescribed features standardised with mean and std of the seed's training split. Reference: E44 free_scaled stage 2 of the same seed. Registered statistic per arm: early = mean over stage-2 epochs 1 to 10 of ln(free_scaled / arm), GMR over seeds with 95% t interval. "Early free advantage removed" if the upper bound is at least 1, "present" if below 1. For E44 prescribed this is computed for reference. Final-epoch GMR and class are reported, not deciding.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; in part A prescribed stage 2 equals stage 1 bit for bit; the encoder is tensor-equal before and after every stage 2; prescribed_gate stage 2 equals E44 prescribed stage 2 bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.
"""

ex = Path("EXPERIMENTS.md")
t = ex.read_text(encoding="utf-8")
assert t.count("### E46.") == 0
assert t.count("### E45.") == 1
i = t.index("### E45.")
j = t.index("\n### ", i + 1)
t = t[:j + 1] + BLOCK + "\n" + t[j + 1:]
idx = list(re.finditer(r"^- \*\*E46\+\*\*:.*$", t, re.M))
assert len(idx) == 1, len(idx)
line = idx[0].group(0).replace("**E46+**", "**E47+**", 1)
t = t[:idx[0].start()] + ("- **E46**: low data (Г-a) and the basis of the prescribed latent (Г-b) on the E44 protocol, PRE-REGISTERED.\n"
                          + line) + t[idx[0].end():]
assert "\u2014" not in BLOCK
ex.write_text(t, encoding="utf-8")
print("E46 block inserted; index updated")
