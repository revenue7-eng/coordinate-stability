#!/usr/bin/env python3
# marker: E46 v1
"""E46 analysis. Run from the repository root. Prints and writes
E46_low_data_basis/results/analysis.json."""
import json, math
from pathlib import Path

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
C46 = Path("E46_low_data_basis/results/cells")
C44 = Path("E44_common_target/results/cells")
T = {9: 2.2622}


def ci(L):
    n = len(L); m = sum(L) / n
    sd = (sum((x - m) ** 2 for x in L) / (n - 1)) ** .5
    h = T[n - 1] * sd / n ** .5
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def cls(lo, hi):
    if lo >= 10: return "O1"
    if 1 < lo and hi < 10: return "O2"
    if 1 < lo < 10 <= hi: return "O3"
    if lo <= 1 <= hi: return "O4"
    if hi < 1: return "O5"
    return "?"


def early(hf, hp):
    return sum(math.log(hf[k] / hp[k]) for k in range(10)) / 10


def conv(h):
    return (sum(h[25:30]) / 5) / (sum(h[20:25]) / 5)


c46 = {s: json.loads((C46 / f"seed_{s}.json").read_text()) for s in SEEDS}
c44 = {s: json.loads((C44 / f"seed_{s}.json").read_text()) for s in SEEDS}
out = {"gates": {}, "lowdata": {}, "basis": {}}

g = out["gates"]
g["prereg_single"] = len({c46[s]["prereg"] for s in SEEDS}) == 1
g["prescribed_s2_equals_s1_lowdata"] = all(
    c46[s]["lowdata"][n]["prescribed"]["s2_hist"] == c46[s]["lowdata"][n]["prescribed"]["s1_hist"]
    for s in SEEDS for n in ("25", "50"))
g["encoders_unchanged"] = all(
    c46[s]["lowdata"][n][a]["encoder_unchanged_in_stage2"] for s in SEEDS for n in ("25", "50")
    for a in ("prescribed", "free_scaled")) and all(
    c46[s]["basis"][a]["encoder_unchanged_in_stage2"] for s in SEEDS
    for a in ("prescribed_gate", "prescribed_sincos", "prescribed_std"))
g["gate_arm_equals_e44"] = all(c46[s]["basis"]["prescribed_gate"]["equals_e44"] for s in SEEDS)
valid = all(g.values())

print("gates:", g, "-> VALID" if valid else "-> VOID")

# Г-a
ref = [math.log(c44[s]["arms"]["free_scaled"]["s2_final"] / c44[s]["arms"]["prescribed"]["s2_final"]) for s in SEEDS]
rows = [("200 (E44)", ref, None)]
for n in ("50", "25"):
    L_ = [math.log(c46[s]["lowdata"][n]["free_scaled"]["s2_final"] / c46[s]["lowdata"][n]["prescribed"]["s2_final"]) for s in SEEDS]
    cv = sorted(conv(c46[s]["lowdata"][n][a]["s2_hist"]) for s in SEEDS for a in ("prescribed", "free_scaled"))
    rows.append((n, L_, cv[len(cv) // 2]))
for n, L_, cvm in rows:
    m, lo, hi = ci(L_)
    k = cls(lo, hi)
    prov = "" if cvm is None else (" PROVISIONAL" if cvm < 0.95 else "")
    out["lowdata"][n] = {"gmr": m, "lo": lo, "hi": hi, "class": k + prov, "median_conv": cvm}
    print(f"Г-a episodes {n}: GMR free_scaled/prescribed {m:.3f} [{lo:.3f}, {hi:.3f}] {k}{prov}"
          + ("" if cvm is None else f"  median conv {cvm:.3f}"))

# Г-b
hfree = {s: c44[s]["arms"]["free_scaled"]["s2_hist"] for s in SEEDS}
for arm, src in (("prescribed (E44)", None), ("prescribed_sincos", "prescribed_sincos"), ("prescribed_std", "prescribed_std")):
    hp = {s: (c44[s]["arms"]["prescribed"]["s2_hist"] if src is None else c46[s]["basis"][src]["s2_hist"]) for s in SEEDS}
    e = ci([early(hfree[s], hp[s]) for s in SEEDS])
    f = ci([math.log(hfree[s][-1] / hp[s][-1]) for s in SEEDS])
    verdict = "early free advantage present" if e[2] < 1 else "early free advantage removed"
    out["basis"][arm] = {"early_gmr": e, "final_gmr": f, "final_class": cls(f[1], f[2]), "early_verdict": verdict}
    print(f"Г-b {arm}: early GMR free/arm {e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}] -> {verdict}; "
          f"final {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}] {cls(f[1], f[2])}")

out["valid"] = valid
Path("E46_low_data_basis/results/analysis.json").write_text(json.dumps(out, indent=1))
