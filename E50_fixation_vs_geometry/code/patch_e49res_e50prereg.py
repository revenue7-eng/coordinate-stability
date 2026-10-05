#!/usr/bin/env python3
# marker: E50 v1
import json, re
from pathlib import Path

F = "\u0424"
A = json.loads(Path("E49_completeness/results/analysis.json").read_text())
assert A["valid"] is True
f3 = lambda v: f"{v:.3f}"
pr = lambda d: ", ".join(f"{k} {f3(v)}" for k, v in d.items())
iv = lambda d: f"{f3(d['gmr'])} [{f3(d['lo'])}, {f3(d['hi'])}], class {d['class']}"
je = A["jepa_over_e2e"]

ev = Path("EVIDENCE.md"); t = ev.read_text(encoding="utf-8")
ex = Path("EXPERIMENTS.md"); x = ex.read_text(encoding="utf-8")
nums = [int(n) for n in re.findall(r"^## " + F + r"(\d+)", t, re.M)]
new = max(nums) + 1
assert new == 86, new

entry = f"""
## {F}{new}: E49, the frozen JEPA latent is complete but nonlinearly laid out; the registered verdict is fixation, with two named limits

- Gates all true: one pre-registration commit; JEPA stage 1 equals E45 bit for bit on every seed; on prescribed_std the linear probe R2 of the four positions is at least 0.999. [verified: E49_completeness/results/analysis.json]
- Probes of the frozen JEPA latent of free_scaled, median over 10 seeds. MLP R2: {pr(A['probes']['jepa']['mlp_r2'])}. Linear R2: {pr(A['probes']['jepa']['linear_r2'])}. Completeness as registered: {A['jepa_completeness']['verdict']} (minimum median MLP R2 {f3(A['jepa_completeness']['min_median_mlp_r2'])}). [verified: analysis.json]
- free_e2e / prescribed_std on final: {iv(A['free_e2e'])}. Frozen JEPA (E48) / prescribed_std: {iv(A['free_scaled JEPA frozen (E48)'])}. Frozen JEPA / free_e2e: {f3(je[0])} [{f3(je[1])}, {f3(je[2])}]. [verified: analysis.json]
- Registered verdict: {A['verdict']}.
- LIMITS, named after the data: (1) free_e2e had 90 epochs in total against 90 + 90 for frozen JEPA, and kept the SIGReg term; that it loses even to frozen JEPA points to under-training, so it is a weak upper bound for a learned encoder. (2) The design does not separate fixed coordinates from a simple (affine, well conditioned) layout of the state: prescribed_std is both. The probes show the learned latent differs from it in layout, not in information.
- INTERPRETATION: the advantage is not a loss of information in the learned latent. Whether it is fixation as such (Г-k) or the simple geometry of the state in the latent (Г-l) is open; {F}39 (a random fixed linear basis matches prescribed) is consistent with either.
- NEXT: E50, a fixed but nonlinearly warped complete encoder, and the learned encoder end to end on an equal budget without SIGReg.
"""
t = t.rstrip("\n") + "\n" + entry

assert x.count("### E49.") == 1 and x.count("### E50.") == 0
i = x.index("### E49."); j = x.index("\n### ", i + 1)
res = f"Result: all gates true. Latent {A['jepa_completeness']['verdict']}; free_e2e {A['free_e2e']['class']}. Registered verdict: fixation, with two named limits. See {F}{new}.\n"
x = x[:j + 1] + res + x[j + 1:]

BLOCK = f"""### E50. Fixation or geometry: a fixed, nonlinearly warped, complete encoder on the E48 target (synthetic Push-T): PRE-REGISTERED

Question. {F}{new}: is the advantage of prescribed_std that its coordinates are fixed (Г-k) or that the state is laid out simply in them (Г-l)?

Arms, 200 episodes, E48 target, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. fixed_warped: frozen encoder f -> f Q1 -> h + A tanh(B h) -> Q2 -> h + A tanh(B h) -> Q3, then standardised on the training split; f the standardised prescribed features; A = 1.5, B = 2 (each step strictly monotone, so the map is invertible); Q1, Q2, Q3 random orthogonal from a private generator seeded 50000 + seed. Fixed like prescribed_std, complete, nonlinearly laid out like the learned latent. Stage 2 only, 90 epochs, E44 loop with SIGReg weight 0.09. Probed as in E49.
2. free_e2e_fair: FreeEncoderScaled trained end to end on the target for 180 epochs (the total budget of frozen JEPA), SIGReg weight 0, otherwise the E44 loop.
prescribed_std and frozen JEPA free_scaled are taken from E48 cells (same seeds, data, target).
Code: E50_fixation_vs_geometry/code/run_e50.py (imports e44_lib.py unchanged), analyze_e50.py.

Registered statistics: GMR fixed_warped / prescribed_std and GMR frozen JEPA (E48) / fixed_warped on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: fixation (Г-k) if fixed_warped / prescribed_std is O4 and frozen JEPA / fixed_warped is O1, O2 or O3; geometry (Г-l) if fixed_warped / prescribed_std is O1, O2 or O3 and frozen JEPA / fixed_warped is O4 or O5; otherwise mixed. The verdict is not deciding if the warp is too weak: minimum over the four positions of the median linear probe R2 of fixed_warped is at least 0.99.
Named, not deciding: GMR free_e2e_fair / prescribed_std and / frozen JEPA; linear and MLP probes of fixed_warped; median convergence ratios.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the fixed_warped encoder is tensor-equal before and after training; fixed_warped is complete (minimum median MLP probe R2 at least 0.99).

Cost: not estimated in advance; the monitor reports the measured rate.
"""
i = x.index("### E49."); j = x.index("\n### ", i + 1)
x = x[:j + 1] + BLOCK + "\n" + x[j + 1:]

l49 = list(re.finditer(r"^- \*\*E49\*\*:.*$", x, re.M)); assert len(l49) == 1
x = x[:l49[0].start()] + f"- **E49**: completeness of the JEPA latent and an end-to-end learned encoder, COMPLETE: latent complete, verdict fixation with named limits ({F}{new})." + x[l49[0].end():]
l50 = list(re.finditer(r"^- \*\*E50\+\*\*:.*$", x, re.M)); assert len(l50) == 1
line = l50[0].group(0).replace("**E50+**", "**E51+**", 1)
x = x[:l50[0].start()] + ("- **E50**: fixation or geometry (fixed warped encoder; fair end-to-end learned encoder), PRE-REGISTERED.\n" + line) + x[l50[0].end():]

for s_ in (entry, res, BLOCK):
    assert "\u2014" not in s_
ev.write_text(t, encoding="utf-8"); ex.write_text(x, encoding="utf-8")
print(f"{F}{new} appended; E49 result; E50 block and index")
