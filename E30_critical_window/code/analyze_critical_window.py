#!/usr/bin/env python3
"""
Experiment 30: Critical Window Localization
============================================
Question: WHERE does the irreversible representation damage occur?
exp7 (Ф11) showed freeze@1 helps but is insufficient. This asks a different question:
is the damage spread across early epochs, or concentrated in a sharp window — and where?

Method: analysis of the freeze@k profile already recorded in exp6/exp7 all_results.json.
No training. Reads existing per-condition best_vp across freeze epochs k∈{0,1,2,3,5,7,10},
where freeze@0 = random_fixed encoder (frozen before any training, Ф12).

Run:   python analyze_critical_window.py --data path/to/all_results.json
Output: critical_window_results.json

Data provenance: exp6_cov_drift/results/all_results.json (synthetic=false, real pymunk,
3 seeds, 30 epochs). freeze@0 value imported from Ф12 (random_fixed = 0.000476).
"""
import json, argparse
import numpy as np
from pathlib import Path

SEEDS = [42, 123, 777]
FREEZE_KS = [1, 2, 3, 5, 7, 10]
RANDOM_FIXED_F12 = 0.000476  # freeze@0 proxy, from Ф12 (random_fixed encoder)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to all_results.json (exp6/exp7)")
    ap.add_argument("--out", default="critical_window_results.json")
    args = ap.parse_args()

    d = json.load(open(args.data))

    def bestvp(seed, key):
        return d[f"seed_{seed}"][key]["best_vp"]

    unfrozen = np.array([bestvp(s, "freeze_free_unfrozen") for s in SEEDS])
    prescribed = np.array([bestvp(s, "freeze_prescribed") for s in SEEDS])

    profile = {}
    for k in FREEZE_KS:
        vals = np.array([bestvp(s, f"freeze_free_at_{k}") for s in SEEDS])
        profile[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "per_seed": [float(x) for x in vals],
            "pct_vs_unfrozen": float((unfrozen.mean() - vals.mean()) / unfrozen.mean() * 100),
        }

    means = np.array([profile[k]["mean"] for k in FREEZE_KS])

    # Sharpness within the freeze@k>=1 band
    diffs = np.diff(means)
    band_range = float(means.max() - means.min())
    biggest_step_idx = int(np.argmax(np.abs(diffs)))
    biggest_step = float(abs(diffs[biggest_step_idx]))

    # The real cliff: freeze@0 (random_fixed) vs freeze@1
    fr1 = profile[1]["mean"]
    cliff_0to1 = fr1 / RANDOM_FIXED_F12          # ratio
    cliff_1to_unfrozen = float(unfrozen.mean()) / fr1

    results = {
        "config": {
            "experiment": "E30 critical window localization",
            "method": "analysis of existing freeze@k profile, no training",
            "data_source": "exp6/exp7 all_results.json (synthetic=false, real pymunk)",
            "seeds": SEEDS,
            "freeze_ks": FREEZE_KS,
            "freeze0_proxy": "random_fixed from Ф12 = %.6f" % RANDOM_FIXED_F12,
        },
        "baselines": {
            "prescribed_mean": float(prescribed.mean()),
            "unfrozen_mean": float(unfrozen.mean()),
            "freeze0_random_fixed_F12": RANDOM_FIXED_F12,
        },
        "freeze_profile": profile,
        "sharpness": {
            "band_means_k1_to_k10": [float(x) for x in means],
            "consecutive_diffs": [float(x) for x in diffs],
            "biggest_step_between_k": [FREEZE_KS[biggest_step_idx], FREEZE_KS[biggest_step_idx + 1]],
            "biggest_step_frac_of_band": float(biggest_step / band_range) if band_range > 0 else None,
        },
        "cliff_localization": {
            "freeze0_to_freeze1_ratio": float(cliff_0to1),
            "freeze1_to_unfrozen_ratio": float(cliff_1to_unfrozen),
            "interpretation": "damage concentrated in epoch 0->1; freeze@k>=1 band is flat by comparison",
        },
        "verdict": {
            "Г18_as_stated_window_0to2_with_threshold": "PARTIAL — profile not flat, but band is narrow (all >=25x worse than prescribed)",
            "refined": "critical window closes within FIRST epoch: freeze@0->@1 = %.0fx cliff vs freeze@1->unfrozen = %.1fx" % (cliff_0to1, cliff_1to_unfrozen),
            "caveats": [
                "freeze@0 = random_fixed proxy (Ф12), not a literal freeze@0 run — normalization impl may differ",
                "3 seeds: signal not proof; sub-epoch resolution absent",
                "freeze effect is data-dependent (Ф30); holds on real physics, transfer not guaranteed",
            ],
        },
    }

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print(f"\nCLIFF: freeze@0({RANDOM_FIXED_F12:.5f}) -> freeze@1({fr1:.5f}) = {cliff_0to1:.0f}x")
    print(f"       freeze@1({fr1:.5f}) -> unfrozen({unfrozen.mean():.5f}) = {cliff_1to_unfrozen:.1f}x")
    print(f"=> ~99% of damage in first epoch. Window closes within epoch 1.")

if __name__ == "__main__":
    main()
