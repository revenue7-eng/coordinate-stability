#!/usr/bin/env python3
# marker: E45 prereg patch v1
"""Writes the E45 pre-registration into EXPERIMENTS.md, updates the index line
and the Ф78 NEXT line. Every substitution is guarded by an exact count."""
from pathlib import Path

BLOCK = """### E45. E44 at 90 epochs, prescribed against free_scaled: PRE-REGISTERED

Question (Ф78). E44 classed free_scaled against prescribed as O4 but PROVISIONAL: both arms were still improving at epoch 30 (median convergence 0.863 and 0.873, below 0.95). Does the class hold when both stages are trained to convergence?

Setup. E44 unchanged except: arms prescribed and free_scaled only; 90 epochs in stage 1 and in stage 2. Library E44_common_target/code/e44_lib.py, unchanged since 483f39e; run_arm is called with epochs=90. Same seeds (42, 123, 777, 1001 to 1007), same data, split and loop. Workers run with OMP_NUM_THREADS=1, as E44 did; every seed file records the thread count.

Registered statistic and outcomes: as E44. Per seed L = ln(final2(free_scaled) / final2(prescribed)) at epoch 90; the class O1 to O5 from the 95% t interval (df 9) of GMR = exp(mean L) decides; the p value is reported only.

Convergence. conv = mean(stage-2 loss over epochs 81 to 90) / mean(epochs 71 to 80). If the median over seeds of conv for either arm is below 0.95, the class is reported as PROVISIONAL.

Validity gates, the campaign is void if any fails: prescribed stage 2 equals stage 1 bit for bit on every seed; the encoder state is unchanged by stage 2 in every cell; the first 30 epochs of stage 1 equal the E44 stage-1 history bit for bit for both arms on every seed (same code, data, seed and thread count, so any difference means the run is not what it claims to be); one pre-registration commit, ancestor of HEAD.

Named, not deciding: the registered statistic on best instead of final, with its class; per arm, E44 final at 30 epochs over E45 final at 90 epochs, as GMR with 95% interval (how much the extra training bought).

Scope: dim 5, E28 synthetic dynamics. If the class is O4 or O2 without PROVISIONAL, Ф78's reading stands as a programme fact for dim 5; if O1 or O3, Ф78 is corrected.

Code: E45_long_training/code/run_e45.py, analyze_e45.py. Results: E45_long_training/results/cells/seed_<s>.json, analysis.json.

Cost: E44 took 4 h 21 min wall for 720 epochs on the three-seed workers. E45 has 2 arms x 2 stages x 90 = 360 epochs per seed, 1080 on a three-seed worker: about 6.5 h wall on four workers."""

ex = Path("EXPERIMENTS.md")
e = ex.read_text(encoding="utf-8")
assert e.count("### E45.") == 0, "E45 block already present"
lines = e.split("\n")
i44 = [i for i, l in enumerate(lines) if l.startswith("### E44.")]
assert len(i44) == 1, i44
j = next((k for k in range(i44[0] + 1, len(lines))
          if lines[k].startswith("### ") or lines[k].startswith("## ")), len(lines))
ins = ([""] if lines[j - 1].strip() else []) + BLOCK.split("\n") + [""]
lines[j:j] = ins
e = "\n".join(lines)
old_idx = ("- **E45+**: free. The nearest candidates are the n=30 replication of R2_readout "
           "on data seed 123 (Ф75 NEXT) and ECA / epiplexity (Г17).")
assert e.count(old_idx) == 1, ("index", e.count(old_idx))
e = e.replace(old_idx,
              "- **E45**: E44 at 90 epochs, prescribed against free_scaled (Ф78), PRE-REGISTERED.\n"
              "- **E46+**: free. The nearest candidates are the n=30 replication of R2_readout "
              "on data seed 123 (Ф75 NEXT) and ECA / epiplexity (Г17).", 1)
assert e.count("### E45.") == 1
ex.write_text(e, encoding="utf-8")

ev = Path("EVIDENCE.md")
t = ev.read_text(encoding="utf-8")
old = "- NEXT: lifting PROVISIONAL needs the same design with more epochs; not yet decided."
assert t.count(old) == 1, ("EVIDENCE", t.count(old))
t = t.replace(old, "- NEXT: E45, pre-registered in EXPERIMENTS.md: prescribed and free_scaled at 90 epochs per stage.", 1)
ev.write_text(t, encoding="utf-8")
print(f"patched: E45 block at line {j + 1}, index, Ф78 NEXT")
