#!/usr/bin/env python3
# marker: E49 v1
import json, re
from pathlib import Path

F = "\u0424"
A = json.loads(Path("E48_nonlinear_target/results/analysis.json").read_text())
assert A["valid"] is True
S, D = A["sizes"], A["low_data_difference"]
f3 = lambda v: f"{v:.3f}"
iv = lambda d: f"{f3(d['gmr'])} [{f3(d['lo'])}, {f3(d['hi'])}], class {d['class']}"

ev = Path("EVIDENCE.md"); t = ev.read_text(encoding="utf-8")
ex = Path("EXPERIMENTS.md"); x = ex.read_text(encoding="utf-8")
nums = [int(n) for n in re.findall(r"^## " + F + r"(\d+)", t, re.M)]
new = max(nums) + 1
assert new == 85, new

entry = f"""
## {F}{new}: E48, the advantage of a standardised fixed basis survives a target nonlinear in the fixed coordinates

- Gates all true: one pre-registration commit; encoders unchanged in every stage 2; free_scaled stage 1 at 200 episodes equals E45 bit for bit on every seed. [verified: E48_nonlinear_target/results/analysis.json]
- Target g(s_t+3) = (agent-block distance, sin theta, cos theta, agent position in the block frame). GMR free_scaled / prescribed_std on final2, n = 10, 90 epochs: 200 episodes {iv(S['200'])}; 25 episodes {iv(S['25'])}. [verified: analysis.json]
- Г-i as registered (class at 200 is O1, O2 or O3): supported. Paired ln-ratio(25) minus ln-ratio(200): {D['mean']:+.3f} [{D['lo']:+.3f}, {D['hi']:+.3f}], {D['verdict']}. [verified: analysis.json]
- Named: GMR on the nonlinear target divided by the E47 GMR, 200 episodes {f3(S['200']['ratio_to_e47_linear'])}, 25 episodes {f3(S['25']['ratio_to_e47_linear'])}. [verified: analysis.json]
- INTERPRETATION: the linear availability of the target named in {F}84 does not carry the advantage; the advantage is as large or larger when the predictor must compute a nonlinear function of the fixed coordinates. The low-data increase of {F}84 is not reproduced as resolved on this target. Synthetic dynamics, dim 5, fully observed state.
- NEXT: E49, whether the frozen JEPA latent of free_scaled loses state information (completeness) and whether the same encoder trained end to end on the target closes the gap.
"""
t = t.rstrip("\n") + "\n" + entry

assert x.count("### E48.") == 1 and x.count("### E49.") == 0
i = x.index("### E48."); j = x.index("\n### ", i + 1)
res = (f"Result: all gates true. 200 episodes {S['200']['class']}, 25 {S['25']['class']}; low-data difference {D['verdict']}. "
       f"Г-i supported as registered. See {F}{new}.\n")
x = x[:j + 1] + res + x[j + 1:]

BLOCK = f"""### E49. Completeness of the JEPA latent and an end-to-end learned encoder on the E48 target (synthetic Push-T): PRE-REGISTERED

Question. {F}{new}: a standardised fixed basis beats the frozen JEPA latent of free_scaled on a nonlinear target. Is that because the JEPA latent loses state information (Г-j, completeness), or because the coordinates are fixed (Г-k)?

Design, 200 episodes, 90 epochs, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. free_scaled stage 1 (JEPA on its own latent), identical to E45 and E48 stage 1. Its frozen latent is probed for the state at the same step, targets (x_a, y_a, x_b, y_b)/512, sin theta, cos theta: linear probe (least squares with bias) and MLP probe (5-64-64-6, 100 epochs, Adam 1e-3), fitted on the training split, R2 per target on the validation split. The same probes on prescribed_std are a sanity check.
2. free_e2e: FreeEncoderScaled trained end to end on the E48 target from the start (encoder not frozen, no JEPA stage), same loop and SIGReg weight.
prescribed_std and the frozen JEPA free_scaled on the same target are taken from E48 cells (same seeds, data and code path).
Code: E49_completeness/code/run_e49.py (imports e44_lib.py unchanged), analyze_e49.py.

Registered statistics. Completeness: per target the median over seeds of the MLP probe R2 on the JEPA latent; "incomplete" if the minimum over the six targets is below 0.95, "complete" if at least 0.99, otherwise unresolved. End to end: GMR free_e2e / prescribed_std on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: completeness (Г-j) if incomplete and free_e2e is O4 or O5; fixation (Г-k) if complete and free_e2e is O1, O2 or O3; otherwise mixed, reported as such.
Named, not deciding: linear probe R2; probes of the end-to-end encoder; GMR of frozen JEPA free_scaled over free_e2e.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; JEPA stage 1 equals E45 free_scaled s1_hist bit for bit on every seed; on prescribed_std the linear probe R2 of the four positions is at least 0.999 on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.
"""
i = x.index("### E48."); j = x.index("\n### ", i + 1)
x = x[:j + 1] + BLOCK + "\n" + x[j + 1:]

l48 = list(re.finditer(r"^- \*\*E48\*\*:.*$", x, re.M)); assert len(l48) == 1
x = x[:l48[0].start()] + (f"- **E48**: E47 on a nonlinear target, COMPLETE: O2 at 200 and 25 episodes ({F}{new}).") + x[l48[0].end():]
l49 = list(re.finditer(r"^- \*\*E49\+\*\*:.*$", x, re.M)); assert len(l49) == 1
line = l49[0].group(0).replace("**E49+**", "**E50+**", 1)
x = x[:l49[0].start()] + ("- **E49**: completeness of the JEPA latent and an end-to-end learned encoder, PRE-REGISTERED.\n" + line) + x[l49[0].end():]

for s_ in (entry, res, BLOCK):
    assert "\u2014" not in s_
ev.write_text(t, encoding="utf-8"); ex.write_text(x, encoding="utf-8")
print(f"{F}{new} appended; E48 result; E49 block and index")
