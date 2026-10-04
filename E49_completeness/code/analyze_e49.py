#!/usr/bin/env python3
# marker: E49 v1
"""E49 analysis. Run from the repository root. Prints and writes
E49_completeness/results/analysis.json."""
import json, math
from pathlib import Path

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
C49 = Path("E49_completeness/results/cells")
C48 = Path("E48_nonlinear_target/results/cells")
T9 = 2.2622
NAMES = ["x_a", "y_a", "x_b", "y_b", "sin", "cos"]


def ci(v):
    n = len(v); m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** .5
    h = T9 * sd / n ** .5
    return m, m - h, m + h


def cls(lo, hi):
    if lo >= 10: return "O1"
    if 1 < lo and hi < 10: return "O2"
    if 1 < lo < 10 <= hi: return "O3"
    if lo <= 1 <= hi: return "O4"
    if hi < 1: return "O5"
    return "?"


def med(v):
    v = sorted(v); return (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


c = {s: json.loads((C49 / f"seed_{s}.json").read_text()) for s in SEEDS}
e = {s: json.loads((C48 / f"seed_{s}.json").read_text()) for s in SEEDS}
out = {"gates": {}, "probes": {}}
g = out["gates"]
g["prereg_single"] = len({c[s]["prereg"] for s in SEEDS}) == 1
g["jepa_s1_equals_e45"] = all(c[s]["jepa"]["s1_equals_e45"] for s in SEEDS)
g["prescribed_std_linear_probe_ge_0.999"] = all(min(c[s]["prescribed_std_probes"]["linear_r2"][:4]) >= 0.999 for s in SEEDS)
valid = all(g.values())
print("gates:", g, "-> VALID" if valid else "-> VOID")

for src in ("jepa", "e2e"):
    out["probes"][src] = {}
    for kind in ("linear_r2", "mlp_r2"):
        meds = [med([c[s][src]["probes"][kind][k] for s in SEEDS]) for k in range(6)]
        out["probes"][src][kind] = dict(zip(NAMES, meds))
        print(f"{src:5s} {kind:9s} median R2: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, meds)))

mlp_min = min(out["probes"]["jepa"]["mlp_r2"].values())
completeness = "incomplete" if mlp_min < 0.95 else ("complete" if mlp_min >= 0.99 else "unresolved")
out["jepa_completeness"] = {"min_median_mlp_r2": mlp_min, "verdict": completeness}
print(f"JEPA latent completeness: min median MLP R2 {mlp_min:.3f} -> {completeness}")

pst = {s: e[s]["sizes"]["200"]["prescribed_std"]["s2_final"] for s in SEEDS}
fjp = {s: e[s]["sizes"]["200"]["free_scaled"]["s2_final"] for s in SEEDS}
for name, num in (("free_e2e", {s: c[s]["e2e"]["final"] for s in SEEDS}), ("free_scaled JEPA frozen (E48)", fjp)):
    m, lo, hi = ci([math.log(num[s] / pst[s]) for s in SEEDS])
    G, Glo, Ghi = math.exp(m), math.exp(lo), math.exp(hi)
    out[name] = {"gmr": G, "lo": Glo, "hi": Ghi, "class": cls(Glo, Ghi)}
    print(f"{name} / prescribed_std: GMR {G:.3f} [{Glo:.3f}, {Ghi:.3f}] {cls(Glo, Ghi)}")
m, lo, hi = ci([math.log(fjp[s] / c[s]["e2e"]["final"]) for s in SEEDS])
out["jepa_over_e2e"] = [math.exp(m), math.exp(lo), math.exp(hi)]
print(f"free JEPA frozen / free_e2e: GMR {math.exp(m):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}] (named)")

k = out["free_e2e"]["class"]
if completeness == "incomplete" and k in ("O4", "O5"):
    verdict = "completeness: the JEPA latent loses state and a learned encoder trained on the target catches up"
elif completeness == "complete" and k in ("O1", "O2", "O3"):
    verdict = "fixation: the JEPA latent is complete and even a target-trained learned encoder stays behind"
else:
    verdict = f"mixed: completeness {completeness}, free_e2e class {k}"
out["verdict"] = verdict
out["valid"] = valid
print("VERDICT:", verdict)
Path("E49_completeness/results/analysis.json").write_text(json.dumps(out, indent=1))
