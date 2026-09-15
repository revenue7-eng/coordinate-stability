#!/usr/bin/env python3
"""E42 analysis, as pre-registered in EXPERIMENTS.md (E42 block).

What is fixed before the run and not negotiable here:
  - Pearson and Spearman of each of the four candidates against best_vp over
    the 30 seed-42 points.
  - Bonferroni over the four candidates: alpha = 0.0125. At n=30, power 0.80,
    that resolves |rho| from 0.567 two-sided.
  - Leave-one-out jackknife reported for every candidate, significant or not.
    It is part of the pre-registration, not a reaction to a result.
  - The control (rms_norm) sits outside the correction and is never a hit.
  - The second data seed checks reproduction of the spread. It does NOT test
    any correlation: 8-10 points cannot resolve one, so none is computed here.

Continuity check: initialisations 1..10 at data seed 42 use the E40
parameterisation, so their best_vp must reproduce E40_init_sweep/results/
sweep.json exactly. If they do not, nothing here may be read beside {F}64,
{F}67 or {F}68 and the discrepancy is the result.

numpy only, no scipy, so this runs in the same venv as the sweep.

Usage: python analyze_e42.py
"""
import json
import math
import os

import numpy as np

BASE = "/mnt/d/coordinate-stability"
SWEEP = os.path.join(BASE, "E42_candidate_sweep/results/sweep.json")
E40_SWEEP = os.path.join(BASE, "E40_init_sweep/results/sweep.json")

CANDIDATES = ("eff_rank", "r2_readout", "cond_number", "smoothness")
CONTROL = ("rms_norm",)
ALPHA_RAW = 0.05
ALPHA_BONF = 0.0125
N_PERM = 200000
PERM_SEED = 0


def ranks(x):
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def fisher_p(r, n):
    """Two-sided p from the Fisher z transform."""
    if n < 4 or abs(r) >= 1.0:
        return float("nan")
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2.0))


def perm_p(a, b, n_perm=N_PERM, seed=PERM_SEED):
    """Two-sided Monte Carlo permutation p for Spearman.

    n=30 makes an exact permutation test impossible, so this is Monte Carlo
    with a fixed seed and a stated iteration count, not the exact test used at
    n=10 in {F}67.
    """
    ra, rb = ranks(a), ranks(b)
    obs = abs(pearson(ra, rb))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        if abs(pearson(ra, rng.permutation(rb))) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def jackknife(a, b, labels):
    """Leave-one-out Pearson: range, and the point that moves it most."""
    full = pearson(a, b)
    rows = []
    for i in range(len(a)):
        m = np.ones(len(a), bool)
        m[i] = False
        rows.append((labels[i], pearson(a[m], b[m])))
    worst = max(rows, key=lambda t: abs(t[1] - full))
    lo = min(r for _, r in rows)
    hi = max(r for _, r in rows)
    return full, lo, hi, worst


def load(seed):
    sd = json.load(open(SWEEP))
    cells = {int(k.split("_i")[1]): v
             for k, v in sd["cells"].items() if k.startswith(f"s{seed}_i")}
    keys = sorted(cells)
    return sd, keys, cells


def continuity(cells, keys):
    if not os.path.exists(E40_SWEEP):
        print("E40 sweep.json absent, continuity check skipped")
        return
    e40 = json.load(open(E40_SWEEP))["inits"]
    shared = [k for k in keys if str(k) in e40]
    if not shared:
        print("no overlap with E40 yet")
        return
    bad = [(k, cells[k]["best_vp"], e40[str(k)]["best_vp"])
           for k in shared
           if cells[k]["best_vp"] != e40[str(k)]["best_vp"]]
    print(f"continuity with E40 over inits {shared[0]}..{shared[-1]}: "
          f"{len(shared) - len(bad)}/{len(shared)} bit-exact")
    for k, got, exp in bad:
        print(f"  MISMATCH init {k}: {got!r} vs E40 {exp!r}")
    if bad:
        print("  the sweep is NOT comparable with Ф64, Ф67, Ф68 as it stands")


def main():
    sd, keys, cells = load(sd_seed := json.load(open(SWEEP))["primary_seed"])
    n = len(keys)
    print(f"seed {sd_seed}: {n} cells\n")
    if n == 0:
        return
    continuity(cells, keys)

    bv = np.array([cells[k]["best_vp"] for k in keys])
    print(f"\nbest_vp spread {bv.max()/bv.min():.4f}x  "
          f"CV {bv.std(ddof=1)/bv.mean():.3f}  median {np.median(bv):.8f}")

    if n < 30:
        print(f"\nPARTIAL: the pre-registration fixes n=30. "
              f"Numbers below are not the registered test.")

    print(f"\n{'candidate':>12} {'pearson':>9} {'spearman':>9} {'p_fisher':>10} "
          f"{'p_perm':>9} {'hit@0.0125':>11}")
    res = {}
    for c in CANDIDATES + CONTROL:
        x = np.array([cells[k][c] for k in keys])
        if not np.isfinite(x).all():
            print(f"{c:>12}   non-finite values, skipped")
            continue
        pe, sp = pearson(x, bv), spearman(x, bv)
        pf = fisher_p(pe, n)
        pp = perm_p(x, bv)
        hit = "control" if c in CONTROL else ("yes" if pf < ALPHA_BONF else "no")
        res[c] = (x, pe, sp)
        print(f"{c:>12} {pe:>+9.4f} {sp:>+9.4f} {pf:>10.4f} {pp:>9.5f} "
              f"{hit:>11}")

    print(f"\njackknife, reported for every candidate regardless of "
          f"significance")
    for c in CANDIDATES + CONTROL:
        if c not in res:
            continue
        x, _, _ = res[c]
        full, lo, hi, (lab, val) = jackknife(x, bv, keys)
        print(f"{c:>12} full {full:>+8.4f}  range [{lo:>+8.4f}, {hi:>+8.4f}]  "
              f"most sensitive to init {lab} ({val:+.4f})")

    second = json.load(open(SWEEP)).get("second_seed")
    if second is not None:
        _, k2, c2 = load(second)
        if k2:
            b2 = np.array([c2[k]["best_vp"] for k in k2])
            print(f"\nsecond data seed {second}: {len(k2)} cells, "
                  f"spread {b2.max()/b2.min():.4f}x  "
                  f"CV {b2.std(ddof=1)/b2.mean():.3f}")
            print("  no correlation is computed here: the second seed tests "
                  "reproduction of the spread, not any relation.")


if __name__ == "__main__":
    main()
