#!/usr/bin/env python3
# marker: E47 v1
"""E47 analysis. Run from the repository root. Prints and writes
E47_std_prescribed/results/analysis.json."""
import json, math
from pathlib import Path

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
C47 = Path("E47_std_prescribed/results/cells")
C45 = Path("E45_long_training/results/cells")
T9 = 2.2622


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


def conv(h):
    return (sum(h[85:90]) / 5) / (sum(h[80:85]) / 5)


def med(v):
    v = sorted(v); return v[len(v) // 2]


c47 = {s: json.loads((C47 / f"seed_{s}.json").read_text()) for s in SEEDS}
c45 = {s: json.loads((C45 / f"seed_{s}.json").read_text()) for s in SEEDS}


def hist(s, n, arm):
    if n == "200" and arm == "free_scaled":
        return c45[s]["arms"]["free_scaled"]["s2_hist"]
    return c47[s]["sizes"][n][arm]["s2_hist"]


out = {"gates": {}, "sizes": {}}
g = out["gates"]
g["prereg_single"] = len({c47[s]["prereg"] for s in SEEDS}) == 1
g["encoders_unchanged"] = all(
    c47[s]["sizes"][n][a]["encoder_unchanged_in_stage2"]
    for s in SEEDS for n in ("25", "50", "200") for a in c47[s]["sizes"][n])
g["gate_seed42_equals_e45"] = bool(c47[42].get("gate_equals_e45"))
g["e45_lengths_90"] = all(len(hist(s, "200", "free_scaled")) == 90 for s in SEEDS)
valid = all(g.values())
print("gates:", g, "-> VALID" if valid else "-> VOID")

lr = {}
for n in ("25", "50", "200"):
    lr[n] = [math.log(hist(s, n, "free_scaled")[-1] / hist(s, n, "prescribed_std")[-1]) for s in SEEDS]
    m, lo, hi = ci(lr[n])
    G, Glo, Ghi = math.exp(m), math.exp(lo), math.exp(hi)
    cp = med([conv(hist(s, n, "prescribed_std")) for s in SEEDS])
    cf = med([conv(hist(s, n, "free_scaled")) for s in SEEDS])
    prov = " PROVISIONAL" if min(cp, cf) < 0.95 else ""
    stab = {}
    for e in (60, 90):
        v = [math.log(hist(s, n, "free_scaled")[e - 1] / hist(s, n, "prescribed_std")[e - 1]) for s in SEEDS]
        a_, b_, c_ = ci(v); stab[e] = (math.exp(a_), math.exp(b_), math.exp(c_))
    out["sizes"][n] = {"gmr": G, "lo": Glo, "hi": Ghi, "class": cls(Glo, Ghi) + prov,
                       "median_conv_prescribed_std": cp, "median_conv_free_scaled": cf,
                       "gmr_ep60": stab[60], "gmr_ep90": stab[90]}
    print(f"n={n:>3}: GMR free_scaled/prescribed_std {G:.3f} [{Glo:.3f}, {Ghi:.3f}] {cls(Glo, Ghi)}{prov}"
          f"  conv std {cp:.3f} free {cf:.3f}  | ep60 {stab[60][0]:.3f} [{stab[60][1]:.3f}, {stab[60][2]:.3f}]")

d = [x - y for x, y in zip(lr["25"], lr["200"])]
m, lo, hi = ci(d)
verdict = "larger at 25" if lo > 0 else ("smaller at 25" if hi < 0 else "not resolved")
out["low_data_difference"] = {"mean": m, "lo": lo, "hi": hi, "verdict": verdict}
print(f"ln-ratio(25) - ln-ratio(200), paired: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
out["valid"] = valid
Path("E47_std_prescribed/results/analysis.json").write_text(json.dumps(out, indent=1))
