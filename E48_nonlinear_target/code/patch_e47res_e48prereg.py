#!/usr/bin/env python3
# marker: E48 v1
import json, re
from pathlib import Path

F = "\u0424"
A = json.loads(Path("E47_std_prescribed/results/analysis.json").read_text())
assert A["valid"] is True
S, D = A["sizes"], A["low_data_difference"]
f3 = lambda v: f"{v:.3f}"
iv = lambda d: f"{f3(d['gmr'])} [{f3(d['lo'])}, {f3(d['hi'])}], class {d['class']}"

ev = Path("EVIDENCE.md"); t = ev.read_text(encoding="utf-8")
ex = Path("EXPERIMENTS.md"); x = ex.read_text(encoding="utf-8")
nums = [int(n) for n in re.findall(r"^## " + F + r"(\d+)", t, re.M)]
new = max(nums) + 1
assert new == 84, new

entry = f"""
## {F}{new}: E47, a standardised fixed basis beats a learned encoder on the common target, by more at low data

- Gates all true: one pre-registration commit; encoders unchanged in every stage 2; on seed 42 plain prescribed stage 2 at 200 episodes and 90 epochs equals E45 bit for bit; every E45 free_scaled history has 90 epochs. [verified: E47_std_prescribed/results/analysis.json]
- GMR free_scaled / prescribed_std on final2, n = 10, 90 epochs: 200 episodes {iv(S['200'])}; 50 episodes {iv(S['50'])}; 25 episodes {iv(S['25'])}. Median convergence ratios: 200 std {f3(S['200']['median_conv_prescribed_std'])}, free {f3(S['200']['median_conv_free_scaled'])}; 25 std {f3(S['25']['median_conv_prescribed_std'])}, free {f3(S['25']['median_conv_free_scaled'])}. [verified: analysis.json]
- Г-g as registered (class at 200 is O1, O2 or O3): supported. Г-a' as registered: paired ln-ratio(25) minus ln-ratio(200) {D['mean']:+.3f} [{D['lo']:+.3f}, {D['hi']:+.3f}], {D['verdict']}: supported. [verified: analysis.json]
- Ratio stability, named: GMR at epoch 60 against 90, 200 episodes {f3(S['200']['gmr_ep60'][0])} against {f3(S['200']['gmr'])}; 25 episodes {f3(S['25']['gmr_ep60'][0])} against {f3(S['25']['gmr'])}. [verified: analysis.json]
- Known asymmetry (E44 block): the common target is the prescribed features of s(t+3), an affine function of the prescribed_std latent, so its predictor receives the target coordinates in its input while free_scaled must also decode them. The comparison is an upper bound in favour of prescribed. Axis alignment alone does not carry it ({F}78, rotated arm 1.00x), linear availability of the target is not excluded.
- INTERPRETATION: with both inputs well scaled, fixing the full state as the latent beats a learned encoder on the same input by about 1.25x at 200 episodes and 1.7x at 25, when the target is linear in the fixed coordinates. The old claim of sample efficiency holds in direction, at tens of percent rather than orders of magnitude. Synthetic dynamics, dim 5, fully observed state.
- NEXT: E48, the same comparison on a target nonlinear in the prescribed coordinates.
"""
t = t.rstrip("\n") + "\n" + entry

assert x.count("### E47.") == 1 and x.count("### E48.") == 0
i = x.index("### E47.")
j = x.index("\n### ", i + 1)
res = (f"Result: all gates true. 200 episodes {S['200']['class']}, 50 {S['50']['class']}, 25 {S['25']['class']}; "
       f"low-data difference {D['verdict']}. Г-g and Г-a' supported as registered. See {F}{new}.\n")
x = x[:j + 1] + res + x[j + 1:]

BLOCK = f"""### E48. E47 on a target nonlinear in the prescribed coordinates (synthetic Push-T): PRE-REGISTERED

Question. {F}{new} names the linear availability of the common target to prescribed_std as not excluded. Does the advantage of a standardised fixed basis survive when the target is nonlinear in the prescribed coordinates?

Target. g(s_{{t+3}}) = (d, sin theta, cos theta, u, v): d the agent-block distance, (u, v) the agent position relative to the block in the block's frame, positions divided by 512. Every component is nonlinear in the prescribed features. Stage 1 (own latent) is unchanged.

Arms and protocol as E47: prescribed_std (stage 2 only), free_scaled (both stages), 90 epochs per stage, at 25 and 200 episodes; free_scaled at 200 is rerun because its stage 2 target changes. Seeds 42, 123, 777, 1001 to 1007 (n = 10). Code: E48_nonlinear_target/code/run_e48.py (imports e44_lib.py unchanged; the world model's external target is replaced in a subclass), analyze_e48.py.

Registered statistic, classes, convergence rule and paired low-data difference exactly as E47. Г-i (the advantage survives a nonlinear target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5.

Named, not deciding: GMR on the nonlinear target divided by the E47 GMR at the same size (the share of the E47 advantage that the linear target carried); GMR at epoch 60.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; free_scaled stage 1 at 200 episodes equals E45 free_scaled s1_hist bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.
"""
i = x.index("### E47.")
j = x.index("\n### ", i + 1)
x = x[:j + 1] + BLOCK + "\n" + x[j + 1:]

l47 = list(re.finditer(r"^- \*\*E47\*\*:.*$", x, re.M)); assert len(l47) == 1
x = x[:l47[0].start()] + (f"- **E47**: standardised prescribed against free_scaled at 25, 50, 200 episodes, COMPLETE: O2 at every size, larger at 25 ({F}{new}).") + x[l47[0].end():]
l48 = list(re.finditer(r"^- \*\*E48\+\*\*:.*$", x, re.M)); assert len(l48) == 1
line = l48[0].group(0).replace("**E48+**", "**E49+**", 1)
x = x[:l48[0].start()] + ("- **E48**: E47 on a nonlinear target, PRE-REGISTERED.\n" + line) + x[l48[0].end():]

for s_ in (entry, res, BLOCK):
    assert "\u2014" not in s_
ev.write_text(t, encoding="utf-8"); ex.write_text(x, encoding="utf-8")
print(f"{F}{new} appended; E47 result; E48 block and index")
