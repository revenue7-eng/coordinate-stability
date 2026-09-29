#!/usr/bin/env python3
# marker: E44 prereg patch v2
"""Writes the E44 pre-registration into EXPERIMENTS.md, updates the index
lines and Ф77. Checks the acceptance log, the determinism rerun and the
acceptance-time library hash first; every text substitution is guarded by an
exact count.

v2: free_raw stage 1 at acceptance matched E28 to a relative 8.7e-7, not bit
for bit. The precondition is now: prescribed bit for bit against E28;
free_raw within a relative 1e-5 of E28; free_raw bit for bit against a second
run in the current environment; prescribed stage 2 equal to stage 1."""
import hashlib
import re
from pathlib import Path

LOG = Path("E44_common_target/results/acceptance.log")
DET = Path("E44_common_target/results/determinism_free_raw.log")
SHA = Path("E44_common_target/results/acceptance_lib.sha256")
LIB = Path("E44_common_target/code/e44_lib.py")

log = LOG.read_text(encoding="utf-8")
mp = re.search(r"^prescribed stage1 best (\S+) E28 (\S+) bit-exact: True ", log, re.M)
assert mp, "prescribed not bit-exact against E28"
mf = re.search(r"^free_raw stage1 best (\S+) E28 (\S+) bit-exact: (True|False) ", log, re.M)
assert mf, "free_raw acceptance line missing"
fv, fe = mf.group(1), mf.group(2)
rel = abs(float(fv) - float(fe)) / float(fe)
assert rel < 1e-5, ("free_raw relative difference to E28", rel)
assert log.count("stage2 hist equals stage1 hist: True") == 1
md = re.search(r"^free_raw stage1 best rerun (\S+) ", DET.read_text(encoding="utf-8"), re.M)
assert md, "determinism rerun line missing"
assert md.group(1) == fv, ("rerun differs from acceptance", md.group(1), fv)
secs = dict(re.findall(r"^(prescribed|free_raw) cell seconds incl\. data ([0-9.]+)$", log, re.M))
assert set(secs) == {"prescribed", "free_raw"}, secs
xp, xf = float(secs["prescribed"]), float(secs["free_raw"])
per = 2 * xp + 4 * xf

sha, size = SHA.read_text().split()
b = LIB.read_bytes()
assert hashlib.sha256(b[:int(size)]).hexdigest() == sha, "acceptance-time library is not a prefix"
assert b"marker: E44 rotated arm v1" in b[int(size):], "rotated block not appended"

COST = (f"Cost, from acceptance.log: prescribed cell (both stages) {xp:.0f} s, free_raw "
        f"stage 1 {xf:.0f} s. Per seed about 2 x {xp:.0f} + 4 x {xf:.0f} = {per:.0f} s "
        f"({per / 60:.0f} min). Ten seeds on four workers (3, 3, 2, 2 seeds): about "
        f"{3 * per / 3600:.1f} h wall if the processes do not slow each other.")

BLOCK = """### E44. Common-target comparison at the E28 dim-5 point (synthetic Push-T): PRE-REGISTERED

Question (Ф77). E28 reports prescribed beating free by 66x at dim 5, on each encoder's own latent and with free on the raw state (Ф76). Is that a difference in quality, or a difference in units and input scaling?

Preconditions, met before this block was committed. At acceptance, e44_lib.py was p2_dim_sweep_full.py byte for byte plus an appended block (results/acceptance_lib.sha256 holds its sha256 and size). The committed e44_lib.py is that file plus a second appended block (the rotated arm). Acceptance (results/acceptance.log): stage 1 reproduced E28 prescribed_dim5_seed42 bit for bit (@PV@). free_dim5_seed42 was reproduced to a relative difference of @REL@ (@FV@ against E28 @FE@), not bit for bit. A second run of free_raw stage 1 with the committed library (results/determinism_free_raw.log) reproduced @FV@ bit for bit, so the current environment is deterministic and the difference from E28 is attributed to the environment E28 ran in, which is not identified [INFERENCE]. Prescribed stage 2 equalled its stage 1 bit for bit. The tolerance for free_raw (relative 1e-5) was set after seeing the acceptance line and is recorded as such; it is five orders of magnitude below the resolution the registered statistic needs. No common-target loss of a free or rotated arm was computed before this commit.

Setup. dim 5, EPISODES 200, EPOCHS 30, data synth(200, seed); split, loaders, optimiser, SIGReg weight 0.09 and loop as in E28.
Arms:
- prescribed: PrescribedEncoder (state divided by its range), as in E28.
- prescribed_rotated: the same features centred, multiplied by a fixed random orthogonal 5x5 matrix (private generator, seed 20260928) and shifted back.
- free_raw: FreeEncoder on the raw state, as in E28.
- free_scaled: the same network on the state divided by (512, 512, 512, 512, 2 pi); same random stream as free_raw.
Stage 1: E28 run_condition at dim 5, own-latent target. Stage 2: encoder frozen, fresh action encoder and predictor, same seed, same loop, target make_prescribed_features(s_{t+3}, 5), a fixed function of the true state.
Seeds: 42, 123, 777 (E28) and 1001 to 1007, n = 10. All arms of a seed share data and split.

Known asymmetry. The common target equals the prescribed latent: the prescribed predictor sees its target coordinates at t..t+2 and has only the dynamics to learn, while a free predictor must also decode them. The comparison is an upper bound in favour of prescribed: O4 or O5 would be strong, O1 weaker than it sounds. prescribed_rotated measures how much of a prescribed advantage is axis alignment with the target.

Registered statistic. Per seed L = ln(final2(free_scaled) / final2(prescribed)), where final2 is the stage-2 validation loss at the last epoch (no selection on validation). Mean L with a 95% t interval (df 9), mapped through exp to the geometric-mean ratio GMR and its interval [lo, hi]. The class alone decides; the t-test p is reported only.

Outcomes:
- O1 order of magnitude survives: lo >= 10.
- O2 prescribed better, not by an order of magnitude: 1 < lo and hi < 10.
- O3 prescribed better, magnitude unresolved: 1 < lo < 10 <= hi.
- O4 no difference resolved: lo <= 1 <= hi.
- O5 free better: hi < 1.
The threshold 10 is the programme's claim "order of magnitude" (Ф36). The E28 figure 66x is a best-epoch loss on each encoder's own latent and is not directly comparable to GMR.

Convergence. For each arm conv = mean(stage-2 loss over epochs 26 to 30) / mean(epochs 21 to 25). If the median over seeds of conv for prescribed or free_scaled is below 0.95, the class is reported as PROVISIONAL (not converged) and is not read as a difference in quality.

Validity gates, the campaign is void if any fails: on every seed prescribed stage 2 equals stage 1 bit for bit; the encoder state is tensor-equal before and after stage 2 in every cell; every seed file carries this pre-registration commit and it is an ancestor of HEAD.

Named, not deciding (each as GMR with its 95% interval over seeds):
- free_raw / free_scaled on final2: the contribution of the input defect (Ф76).
- prescribed_rotated / prescribed on final2: the contribution of axis alignment.
- free_raw / prescribed on stage-1 best: the E28-style unit-confounded ratio, expected near E28.
- the registered statistic on best instead of final, with its class.

Scope: dim 5 and E28's synthetic dynamics only. Ф17 (dim 11) and the other Ф18 dims stay "not interpretable".

Code: E44_common_target/code/e44_lib.py, run_e44.py (runs only if this commit is an ancestor of HEAD, holds this block, touches the three code files, and the code is unchanged since), analyze_e44.py. Results: E44_common_target/results/cells/seed_<s>.json, analysis.json.

@COST@""".replace("@COST@", COST).replace("@PV@", mp.group(1)).replace(
    "@FV@", fv).replace("@FE@", fe).replace("@REL@", f"{rel:.1e}")
assert "@" not in BLOCK, "unfilled placeholder"

ex = Path("EXPERIMENTS.md")
e = ex.read_text(encoding="utf-8")
assert e.count("### E44.") == 0, "E44 block already present"
lines = e.split("\n")
i43 = [i for i, l in enumerate(lines) if l.startswith("### E43.")]
assert len(i43) == 1, i43
j = next((k for k in range(i43[0] + 1, len(lines))
          if lines[k].startswith("### ") or lines[k].startswith("## ")), len(lines))
ins = ([""] if lines[j - 1].strip() else []) + BLOCK.split("\n") + [""]
lines[j:j] = ins
e = "\n".join(lines)
old_idx = ("- **E44+**: free. The nearest candidate is the E28 dim-5 point on a common "
           "external target (Ф77); ECA / epiplexity (Г17) follows.")
assert e.count(old_idx) == 1, ("index", e.count(old_idx))
e = e.replace(old_idx,
              "- **E44**: common-target comparison at the E28 dim-5 point (Ф77), PRE-REGISTERED.\n"
              "- **E45+**: free. The nearest candidates are the n=30 replication of R2_readout "
              "on data seed 123 (Ф75 NEXT) and ECA / epiplexity (Г17).", 1)
assert e.count("### E44.") == 1
ex.write_text(e, encoding="utf-8")

ev = Path("EVIDENCE.md")
t = ev.read_text(encoding="utf-8")
old1 = "E28 uses its own synthetic dynamics, where this is not checked."
new1 = ("[verified: python3 E44_common_target/code/e28_persistence_check.py] On E28's own "
        "synthetic dynamics the block moves on 4.4% to 4.7% of recorded steps (seeds 42, 123, "
        "777). The one-step persistence MSE of the prescribed latent is about 0.000001 at dim 2 "
        "and about 0.004 at dim 5. At dim 2 the E28 prescribed loss (0.000004 to 0.000006) is "
        "above that persistence on all three seeds while the E28 ratio to free is 1539x to "
        "2175x; persistence is over all windows, the E28 loss is the best epoch on a 10% "
        "validation split. Ordered by mean persistence and by mean ratio, the seven dims agree "
        "except for dims 1, 5 and 7 (Spearman -0.89, descriptive).")
old2 = ("- NEXT: E44, the E28 dim-5 point on a common external target, with three arms: "
        "prescribed_5, free on the raw state, free on the range-normalised state.")
new2 = ("- NEXT: E44, pre-registered in EXPERIMENTS.md: four arms at the E28 dim-5 point on a "
        "common external target.")
for o in (old1, old2):
    assert t.count(o) == 1, ("EVIDENCE", o[:40], t.count(o))
t = t.replace(old1, new1, 1).replace(old2, new2, 1)
ev.write_text(t, encoding="utf-8")

print(f"patched: E44 block at line {j + 1}, index, Ф77; per seed {per:.0f} s")
