#!/usr/bin/env python3
# marker: E47 v1
import json, re
from pathlib import Path

F = "\u0424"
A = json.loads(Path("E46_low_data_basis/results/analysis.json").read_text())
assert A["valid"] is True
ld, bs = A["lowdata"], A["basis"]
f3 = lambda x: f"{x:.3f}"
iv = lambda d: f"{f3(d['gmr'])} [{f3(d['lo'])}, {f3(d['hi'])}]"
it = lambda t: f"{f3(t[0])} [{f3(t[1])}, {f3(t[2])}]"

ev = Path("EVIDENCE.md"); t = ev.read_text(encoding="utf-8")
ex = Path("EXPERIMENTS.md"); x = ex.read_text(encoding="utf-8")

nums = [int(n) for n in re.findall(r"^## " + F + r"(\d+)", t, re.M)]
new = max(nums) + 1
assert new == 83, new
for k in (78, 79):
    assert len(re.findall(r"^## " + F + str(k) + r"\b", t, re.M)) == 1, k

entry = f"""
## {F}{new}: E46, plain prescribed loses at low data; standardising the prescribed features removes the early free advantage and puts prescribed ahead at 200 episodes

- Gates all true: one pre-registration commit, prescribed stage 2 equals stage 1 at 25 and 50 episodes, encoders unchanged in every stage 2, prescribed_gate equals E44 prescribed stage 2 bit for bit on every seed. [verified: E46_low_data_basis/results/analysis.json]
- Г-a as registered, GMR free_scaled / plain prescribed on final2, n = 10: 25 episodes {iv(ld['25'])}, class {ld['25']['class']}; 50 episodes {iv(ld['50'])}, class {ld['50']['class']}; 200 episodes (E44) {iv(ld['200 (E44)'])}. By the registered rule (O4 or O5 at 25) Г-a is refuted for range-normalised prescribed. [verified: analysis.json]
- Г-b, early GMR free_scaled / arm over stage-2 epochs 1 to 10 at 200 episodes: plain prescribed (E44) {it(bs['prescribed (E44)']['early_gmr'])}; prescribed_sincos {it(bs['prescribed_sincos']['early_gmr'])}, {bs['prescribed_sincos']['early_verdict']}; prescribed_std {it(bs['prescribed_std']['early_gmr'])}, {bs['prescribed_std']['early_verdict']}. [verified: analysis.json]
- Final at 200 episodes, reported and not deciding (no convergence check registered for part B): prescribed_std {it(bs['prescribed_std']['final_gmr'])}, class {bs['prescribed_std']['final_class']}; prescribed_sincos {it(bs['prescribed_sincos']['final_gmr'])}, class {bs['prescribed_sincos']['final_class']}. [verified: analysis.json]
- INTERPRETATION: the early free advantage recorded as EXPLORATORY under {F}79 is the scale of the range-normalised prescribed features (values in [0, 1], small variance), not the wrap of the angle. This is the mirror of {F}76 on the prescribed side [INFERENCE, from the std arm; mechanism not isolated further]. E44, E45 and the Г-a verdict above compare a poorly scaled prescribed latent with a well scaled free one; their classes do not test fixation as such.
- NEXT: E47, standardised prescribed against free_scaled at 25, 50 and 200 episodes, 90 epochs.
"""
status = f"\n- STATUS ({F}{new}): the prescribed arm here is range-normalised, not standardised; its scale handicaps the predictor. Does not test fixation as such."
for k in (79, 78):
    m = re.search(r"^## " + F + str(k) + r"\b.*$", t, re.M)
    t = t[:m.end()] + status + t[m.end():]
t = t.rstrip("\n") + "\n" + entry

# EXPERIMENTS: E46 result, E47 block, index
assert x.count("### E46.") == 1 and x.count("### E47.") == 0
i = x.index("### E46.")
j = x.index("\n### ", i + 1)
res46 = (f"Result: all gates true. Г-a: 25 episodes {ld['25']['class']}, 50 episodes {ld['50']['class']} (plain prescribed); refuted as registered. "
         f"Г-b: prescribed_std {bs['prescribed_std']['early_verdict']}, prescribed_sincos {bs['prescribed_sincos']['early_verdict']}. "
         f"See {F}{new}.\n")
x = x[:j + 1] + res46 + x[j + 1:]

BLOCK = f"""### E47. Standardised prescribed against free_scaled on the common target at 25, 50 and 200 episodes (synthetic Push-T): PRE-REGISTERED

Question. {F}{new}: with the prescribed features standardised, is there a prescribed advantage on the common target, and is it larger at low data?

Arms. prescribed_std: the five prescribed features standardised with mean and std of the training split of that seed and size; encoder fixed, stage 2 only (as E46 part B). free_scaled: both stages as E45, 90 epochs each, at 25 and 50 episodes; at 200 episodes taken from E45 cells (same seeds, data and code path). Common target and loop as E44. 90 epochs per stage. Seeds 42, 123, 777, 1001 to 1007 (n = 10). OMP_NUM_THREADS=1 per worker. Fewer episodes at fixed epochs also means fewer optimiser steps; not separated.

Code: E47_std_prescribed/code/run_e47.py (imports e44_lib.py unchanged), analyze_e47.py.

Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed_std), GMR with 95% t interval (df 9), classes O1 to O5 as E44. Convergence: conv = mean(stage-2 loss, epochs 86 to 90) / mean(epochs 81 to 85); PROVISIONAL if the median over seeds for either arm is below 0.95. Г-g (a standardised fixed basis is better on the common target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5. Г-a' (the advantage is larger at low data): paired difference ln-ratio(25) minus ln-ratio(200) over seeds with 95% t interval; "larger at 25" if the lower bound is above 0, "smaller at 25" if the upper bound is below 0, otherwise not resolved.

Named, not deciding: GMR at epoch 60 against epoch 90 per size (ratio stability, {F}79 INTERPRETATION); the 50-episode class.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; on seed 42 plain prescribed stage 2 at 200 episodes and 90 epochs equals E45 prescribed s2_hist bit for bit; every E45 free_scaled history has 90 epochs.

Cost: not estimated in advance; the monitor reports the measured rate.
"""
i = x.index("### E46.")
j = x.index("\n### ", i + 1)
x = x[:j + 1] + BLOCK + "\n" + x[j + 1:]

l46 = list(re.finditer(r"^- \*\*E46\*\*:.*$", x, re.M)); assert len(l46) == 1
x = x[:l46[0].start()] + (f"- **E46**: low data (Г-a) and the basis of the prescribed latent (Г-b), COMPLETE: Г-a refuted for plain prescribed ({ld['25']['class']} at 25); "
                          f"standardising removes the early free advantage ({F}{new}).") + x[l46[0].end():]
l47 = list(re.finditer(r"^- \*\*E47\+\*\*:.*$", x, re.M)); assert len(l47) == 1
line = l47[0].group(0).replace("**E47+**", "**E48+**", 1)
x = x[:l47[0].start()] + ("- **E47**: standardised prescribed against free_scaled at 25, 50, 200 episodes, 90 epochs, PRE-REGISTERED.\n" + line) + x[l47[0].end():]

for s_ in (entry, status, res46, BLOCK):
    assert "\u2014" not in s_
ev.write_text(t, encoding="utf-8"); ex.write_text(x, encoding="utf-8")
print(f"{F}{new} appended; STATUS under {F}78, {F}79; E46 result; E47 block and index")
