#!/usr/bin/env python3
# marker: E50 v1
"""E50 analysis. Run from the repository root. Prints and writes
E50_fixation_vs_geometry/results/analysis.json."""
import json, math
from pathlib import Path

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
C50 = Path("E50_fixation_vs_geometry/results/cells")
C48 = Path("E48_nonlinear_target/results/cells")
T9 = 2.2622
NAMES = ["x_a", "y_a", "x_b", "y_b", "sin", "cos"]


def ci(v):
    n = len(v); m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** .5
    h = T9 * sd / n ** .5
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def cls(lo, hi):
    if lo >= 10: return "O1"
    if 1 < lo and hi < 10: return "O2"
    if 1 < lo < 10 <= hi: return "O3"
    if lo <= 1 <= hi: return "O4"
    if hi < 1: return "O5"
    return "?"


def med(v):
    v = sorted(v); return (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def conv(h):
    return (sum(h[-5:]) / 5) / (sum(h[-10:-5]) / 5)


c = {s: json.loads((C50 / f"seed_{s}.json").read_text()) for s in SEEDS}
e = {s: json.loads((C48 / f"seed_{s}.json").read_text()) for s in SEEDS}
pst = {s: e[s]["sizes"]["200"]["prescribed_std"]["s2_final"] for s in SEEDS}
fjp = {s: e[s]["sizes"]["200"]["free_scaled"]["s2_final"] for s in SEEDS}
fw = {s: c[s]["fixed_warped"]["s2_final"] for s in SEEDS}
fe = {s: c[s]["free_e2e_fair"]["final"] for s in SEEDS}
out = {"gates": {}}

lin = [med([c[s]["fixed_warped"]["probes"]["linear_r2"][k] for s in SEEDS]) for k in range(6)]
mlp = [med([c[s]["fixed_warped"]["probes"]["mlp_r2"][k] for s in SEEDS]) for k in range(6)]
out["warped_probes"] = {"linear_r2": dict(zip(NAMES, lin)), "mlp_r2": dict(zip(NAMES, mlp))}
print("fixed_warped linear R2: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, lin)))
print("fixed_warped MLP    R2: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, mlp)))

g = out["gates"]
g["prereg_single"] = len({c[s]["prereg"] for s in SEEDS}) == 1
g["warped_encoder_unchanged"] = all(c[s]["fixed_warped"]["encoder_unchanged"] for s in SEEDS)
g["warped_complete_mlp_ge_0.99"] = min(mlp) >= 0.99
valid = all(g.values())
curved = min(lin[:4]) < 0.99
out["warped_curved"] = curved
print("gates:", g, "-> VALID" if valid else "-> VOID", "| warped curved (min median linear R2 of positions < 0.99):", curved)

rows = {"fixed_warped / prescribed_std": (fw, pst), "free JEPA frozen (E48) / fixed_warped": (fjp, fw),
        "free_e2e_fair / prescribed_std": (fe, pst), "free_e2e_fair / free JEPA frozen (E48)": (fe, fjp)}
for name, (num, den) in rows.items():
    G, lo, hi = ci([math.log(num[s] / den[s]) for s in SEEDS])
    out[name] = {"gmr": G, "lo": lo, "hi": hi, "class": cls(lo, hi)}
    print(f"{name}: GMR {G:.3f} [{lo:.3f}, {hi:.3f}] {cls(lo, hi)}")
out["conv_median"] = {"fixed_warped": med([conv(c[s]["fixed_warped"]["s2_hist"]) for s in SEEDS]),
                      "free_e2e_fair": med([conv(c[s]["free_e2e_fair"]["hist"]) for s in SEEDS])}
print("median conv:", {k: round(v, 3) for k, v in out["conv_median"].items()})

W = out["fixed_warped / prescribed_std"]["class"]
FW = out["free JEPA frozen (E48) / fixed_warped"]["class"]
if not curved:
    verdict = "not deciding: the warp is too weak (warped latent is nearly linear in the state)"
elif W == "O4" and FW in ("O1", "O2", "O3"):
    verdict = "fixation (Г-k): a fixed warped basis matches prescribed_std and beats the learned latent"
elif W in ("O1", "O2", "O3") and FW in ("O4", "O5"):
    verdict = "geometry (Г-l): a fixed warped basis loses to prescribed_std and is no better than the learned latent"
else:
    verdict = f"mixed: warped/prescribed_std {W}, free/warped {FW}"
out["verdict"] = verdict
out["valid"] = valid
print("VERDICT:", verdict)
Path("E50_fixation_vs_geometry/results/analysis.json").write_text(json.dumps(out, indent=1))
