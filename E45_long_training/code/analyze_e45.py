#!/usr/bin/env python3
# marker: E45 analysis v1
"""E45 analysis, committed with the pre-registration. Reads
E45_long_training/results/cells/seed_<s>.json for the ten registered seeds,
checks the gates, prints the registered class and the convergence check, and
writes E45_long_training/results/analysis.json."""
import json
import math
import subprocess
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
ARMS = ["prescribed", "free_scaled"]
T975_DF9 = 2.2621571627982053
CELLS = Path("E45_long_training/results/cells")
E44_CELLS = Path("E44_common_target/results/cells")
OUT = Path("E45_long_training/results/analysis.json")


def gmr(logs):
    x = np.asarray(logs, float)
    assert len(x) == 10, len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(m / se), 9)) if se > 0 else 0.0
    except ImportError:
        p = None
    return {"gmr": math.exp(m), "lo": math.exp(m - T975_DF9 * se),
            "hi": math.exp(m + T975_DF9 * se), "mean_log": m, "se_log": se, "p": p}


def classify(lo, hi):
    if lo >= 10:
        return "O1"
    if lo > 1 and hi < 10:
        return "O2"
    if lo > 1:
        return "O3"
    if hi < 1:
        return "O5"
    return "O4"


def conv(h):
    return float(np.mean(h[80:90]) / np.mean(h[70:80]))


def main():
    R = {}
    for s in SEEDS:
        f = CELLS / f"seed_{s}.json"
        assert f.exists(), f"missing {f}"
        R[s] = json.loads(f.read_text())
        assert sorted(R[s]["arms"]) == sorted(ARMS), (s, sorted(R[s]["arms"]))
        assert R[s]["epochs"] == 90
    preregs = {r["prereg"] for r in R.values()}
    prereg = preregs.pop() if len(preregs) == 1 else None
    gates = {
        "one_prereg_commit_ancestor_of_HEAD": prereg is not None and subprocess.run(
            ["git", "merge-base", "--is-ancestor", prereg, "HEAD"]).returncode == 0,
        "prescribed_stage2_equals_stage1": all(
            R[s]["arms"]["prescribed"]["s2_hist"] == R[s]["arms"]["prescribed"]["s1_hist"]
            for s in SEEDS),
        "encoders_unchanged_in_stage2": all(
            R[s]["arms"][a]["encoder_unchanged_in_stage2"] for s in SEEDS for a in ARMS),
        "stage1_first30_equals_e44": all(
            R[s]["arms"][a]["stage1_first30_equals_e44"] for s in SEEDS for a in ARMS),
    }
    void = not all(gates.values())
    val = lambda key: {a: {s: R[s]["arms"][a][key] for s in SEEDS} for a in ARMS}
    f2, b2 = val("s2_final"), val("s2_best")
    lr = lambda num, den: [math.log(num[s] / den[s]) for s in SEEDS]
    primary = gmr(lr(f2["free_scaled"], f2["prescribed"]))
    primary["class"] = classify(primary["lo"], primary["hi"])
    cv = {a: float(np.median([conv(R[s]["arms"][a]["s2_hist"]) for s in SEEDS])) for a in ARMS}
    provisional = any(v < 0.95 for v in cv.values())
    on_best = gmr(lr(b2["free_scaled"], b2["prescribed"]))
    on_best["class"] = classify(on_best["lo"], on_best["hi"])
    e44_f2 = {a: {s: json.loads((E44_CELLS / f"seed_{s}.json").read_text())["arms"][a]["s2_final"]
                  for s in SEEDS} for a in ARMS}
    gain = {a: gmr(lr(e44_f2[a], f2[a])) for a in ARMS}
    verdict = "VOID" if void else primary["class"] + (" PROVISIONAL" if provisional else "")
    out = {"prereg": prereg, "gates": gates, "verdict": verdict,
           "primary_free_scaled_over_prescribed_final2": primary,
           "convergence_median_conv_last10_over_prev10": cv, "provisional": provisional,
           "registered_statistic_on_best": on_best,
           "gain_e44_final30_over_e45_final90": gain,
           "per_seed_final2": {a: [f2[a][s] for s in SEEDS] for a in ARMS}, "seeds": SEEDS}
    OUT.write_text(json.dumps(out, indent=1))
    print("gates:", gates)
    print("verdict:", verdict)
    print(f"primary GMR {primary['gmr']:.4g} CI [{primary['lo']:.4g}, {primary['hi']:.4g}] p {primary['p']}")
    print("convergence (median, last 10 over previous 10):", {a: round(v, 4) for a, v in cv.items()})
    print(f"on best: GMR {on_best['gmr']:.4g} CI [{on_best['lo']:.4g}, {on_best['hi']:.4g}] class {on_best['class']}")
    for a, g in gain.items():
        print(f"{a}: E44 final at 30 over E45 final at 90, GMR {g['gmr']:.4g} CI [{g['lo']:.4g}, {g['hi']:.4g}]")
    print("per seed final2:")
    for s in SEEDS:
        print(f"  {s:5d} " + "  ".join(f"{a} {f2[a][s]:.6f}" for a in ARMS))
    print("written", OUT)


if __name__ == "__main__":
    main()
