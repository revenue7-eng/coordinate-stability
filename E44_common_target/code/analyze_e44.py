#!/usr/bin/env python3
# marker: E44 analysis v1
"""E44 analysis, committed with the pre-registration.

Reads E44_common_target/results/cells/seed_<s>.json for the ten registered
seeds, checks the validity gates, prints the registered class, the convergence
check and the named secondary contrasts, and writes
E44_common_target/results/analysis.json.
"""
import json
import math
import subprocess
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
ARMS = ["prescribed", "prescribed_rotated", "free_raw", "free_scaled"]
T975_DF9 = 2.2621571627982053
CELLS = Path("E44_common_target/results/cells")
OUT = Path("E44_common_target/results/analysis.json")


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
            "hi": math.exp(m + T975_DF9 * se), "mean_log": m, "se_log": se,
            "p": p}


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
    return float(np.mean(h[25:30]) / np.mean(h[20:25]))


def logratio(num, den):
    return [math.log(num[s] / den[s]) for s in SEEDS]


def main():
    R = {}
    for s in SEEDS:
        f = CELLS / f"seed_{s}.json"
        assert f.exists(), f"missing {f}"
        R[s] = json.loads(f.read_text())
        assert sorted(R[s]["arms"]) == sorted(ARMS), (s, sorted(R[s]["arms"]))

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
    }
    void = not all(gates.values())

    val = lambda key: {a: {s: R[s]["arms"][a][key] for s in SEEDS} for a in ARMS}
    f2, b2, b1 = val("s2_final"), val("s2_best"), val("s1_best")

    primary = gmr(logratio(f2["free_scaled"], f2["prescribed"]))
    primary["class"] = classify(primary["lo"], primary["hi"])
    cv = {a: float(np.median([conv(R[s]["arms"][a]["s2_hist"]) for s in SEEDS]))
          for a in ARMS}
    provisional = cv["prescribed"] < 0.95 or cv["free_scaled"] < 0.95

    on_best = gmr(logratio(b2["free_scaled"], b2["prescribed"]))
    on_best["class"] = classify(on_best["lo"], on_best["hi"])
    secondary = {
        "input_defect_free_raw_over_free_scaled_final2":
            gmr(logratio(f2["free_raw"], f2["free_scaled"])),
        "axis_alignment_rotated_over_prescribed_final2":
            gmr(logratio(f2["prescribed_rotated"], f2["prescribed"])),
        "e28_style_free_raw_over_prescribed_stage1_best":
            gmr(logratio(b1["free_raw"], b1["prescribed"])),
        "registered_statistic_on_best": on_best,
    }

    verdict = "VOID" if void else primary["class"] + (" PROVISIONAL" if provisional else "")
    out = {"prereg": prereg, "gates": gates, "verdict": verdict,
           "primary_free_scaled_over_prescribed_final2": primary,
           "convergence_median_conv": cv, "provisional": provisional,
           "secondary": secondary,
           "per_seed_final2": {a: [f2[a][s] for s in SEEDS] for a in ARMS},
           "seeds": SEEDS}
    OUT.write_text(json.dumps(out, indent=1))

    print("gates:", gates)
    print("verdict:", verdict)
    print(f"primary GMR {primary['gmr']:.4g} CI [{primary['lo']:.4g}, {primary['hi']:.4g}] p {primary['p']}")
    print("convergence (median conv, stage 2):", {a: round(v, 4) for a, v in cv.items()})
    for k, v in secondary.items():
        print(f"{k}: GMR {v['gmr']:.4g} CI [{v['lo']:.4g}, {v['hi']:.4g}]"
              + (f" class {v['class']}" if "class" in v else ""))
    print("per seed final2:")
    for i, s in enumerate(SEEDS):
        print(f"  {s:5d} " + "  ".join(f"{a} {f2[a][s]:.6f}" for a in ARMS))
    print("written", OUT)


if __name__ == "__main__":
    main()
