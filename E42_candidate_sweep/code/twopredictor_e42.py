#!/usr/bin/env python3
"""Two-predictor control for the E42 candidate sweep.

Reads results/persistence.json written by persistence_e42.py; recomputes it
from checkpoints only if that file is absent or incomplete, so the usual run
needs no forward pass and no torch.

What this adds over persistence_e42.py:
  1. OLS of log(vp) on BOTH log(persistence) and log(rms_norm), which are
     nearly orthogonal to each other, and candidates against that residual.
     rms_norm is the pre-registered control; on the one-predictor residual it
     outscored every candidate, so it has to be controlled before any
     candidate claim survives.
  2. Scale-anomaly probe: persistence would scale as the square of the
     embedding norm if it were a pure scale quantity, so OLS of
     log(persistence) on log(rms_norm) should give a slope near 2.
  3. Log versus raw reconciliation for the exploratory eff_rank numbers
     (-0.2761 at n=30, -0.3439 at n=10) that did not reproduce in logs.

Nothing here is pre-registered. Output is exploratory.
"""
import os
import sys
import json
import math

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(CODE)
sys.path.insert(0, CODE)

SWEEP = os.path.join(EXP, "results", "sweep.json")
PERS = os.path.join(EXP, "results", "persistence.json")
CKPT = os.path.join(EXP, "checkpoints")

GROUPS = [("s42", 42), ("s123", 123)]
CANDIDATES = ["eff_rank", "r2_readout", "cond_number", "smoothness"]
CONTROL = "rms_norm"
NUMERATORS = ["best_vp", "final_vp", "mean_last3"]
BONF = 0.05 / 4


# ---------- statistics, no scipy ----------

def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def fisher_p(r, n, k=0):
    """Two-sided p from Fisher z. k = number of controlled variables."""
    dof = n - 3 - k
    if dof < 1 or not (-1 < r < 1):
        return float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    return math.erfc(abs(z) / math.sqrt(2))


def center(v):
    m = sum(v) / len(v)
    return [a - m for a in v]


def ols1(x, y):
    """Slope and residuals of y on x."""
    cx, cy = center(x), center(y)
    sxx = sum(a * a for a in cx)
    b = sum(a * c for a, c in zip(cx, cy)) / sxx
    return b, [c - b * a for a, c in zip(cx, cy)]


def ols2(x1, x2, y):
    """Slopes, residuals and R2 of y on x1 and x2 together."""
    c1, c2, cy = center(x1), center(x2), center(y)
    s11 = sum(a * a for a in c1)
    s22 = sum(a * a for a in c2)
    s12 = sum(a * b for a, b in zip(c1, c2))
    s1y = sum(a * c for a, c in zip(c1, cy))
    s2y = sum(b * c for b, c in zip(c2, cy))
    det = s11 * s22 - s12 * s12
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    res = [c - b1 * a - b2 * b for a, b, c in zip(c1, c2, cy)]
    syy = sum(c * c for c in cy)
    r2 = 1.0 - sum(r * r for r in res) / syy
    return b1, b2, res, r2


def flag(p):
    if p != p:
        return ""
    if p < BONF:
        return "  <- passes Bonferroni 0.0125"
    if p < 0.05:
        return "  (nominal only)"
    return ""


# ---------- persistence, from file or from checkpoints ----------

def load_persistence(cells):
    if os.path.exists(PERS):
        d = json.load(open(PERS))
        if all(n in d for n in cells):
            print(f"persistence: read {len(d)} cells from {PERS}")
            return d
        print("persistence.json incomplete, recomputing from checkpoints")
    else:
        print("persistence.json absent, computing from checkpoints")
    return compute_persistence(cells)


def compute_persistence(cells):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    from e42_lib import FE, DS, collect_gym_data

    out = {}
    for tag, seed in GROUPS:
        keys = [n for n in cells if n.startswith(tag + "_")]
        if not keys:
            continue
        eps = collect_gym_data(n_ep=200, max_steps=300, fs=5, seed=seed)
        ds = DS(eps, 3)
        nt = int(len(ds) * 0.9)
        _, va = random_split(ds, [nt, len(ds) - nt],
                             generator=torch.Generator().manual_seed(seed))
        vl = DataLoader(va, batch_size=64)
        for n in keys:
            k = int(n.split("_i")[1])
            p = os.path.join(CKPT, f"s{seed}_i{k}_enc_at_freeze.pt")
            if not os.path.exists(p):
                continue
            enc = FE()
            enc.load_state_dict(torch.load(p, map_location="cpu"))
            enc.eval()
            with torch.no_grad():
                tot, m = 0.0, 0
                for s, _a in vl:
                    emb = enc(s)
                    tot += F.mse_loss(emb[:, 2], emb[:, 3]).item() * s.size(0)
                    m += s.size(0)
            out[n] = tot / m
    json.dump(out, open(PERS, "w"), indent=2)
    print(f"wrote {PERS} ({len(out)} cells)")
    return out


# ---------- main ----------

def main():
    cells = json.load(open(SWEEP))["cells"]
    pers_all = load_persistence(cells)

    for tag, _seed in GROUPS:
        keys = sorted((n for n in cells if n.startswith(tag + "_") and n in pers_all),
                      key=lambda n: int(n.split("_i")[1]))
        if not keys:
            continue
        n = len(keys)
        print(f"\n{'=' * 70}\n{tag}   n={n}\n{'=' * 70}")

        lp = [math.log(pers_all[k]) for k in keys]
        lrms = [math.log(cells[k][CONTROL]) for k in keys]
        cand = {c: [cells[k][c] for k in keys] for c in CANDIDATES}

        # --- 2. scale anomaly ---
        b_scale, _ = ols1(lrms, lp)
        r_scale = pearson(lrms, lp)
        print("\nscale anomaly probe")
        print(f"  OLS log(persistence) on log(rms_norm): slope {b_scale:+.4f}"
              f"   (pure scale would give 2.0000)")
        print(f"  corr log(persistence) vs log(rms_norm): {r_scale:+.4f}"
              f" (p {fisher_p(r_scale, n):.4f})")
        print(f"  the two controls are {'nearly orthogonal' if abs(r_scale) < 0.3 else 'entangled'},"
              f" so the two-predictor fit is {'well' if abs(r_scale) < 0.3 else 'poorly'} conditioned")

        for num in NUMERATORS:
            v = [cells[k][num] for k in keys]
            lv = [math.log(a) for a in v]
            ratio = [a / pers_all[k] for a, k in zip(v, keys)]
            lr = [math.log(a) for a in ratio]

            b1, res1 = ols1(lp, lv)
            b1b, b2b, res2, r2_2 = ols2(lp, lrms, lv)
            r2_1 = pearson(lv, lp) ** 2

            print(f"\n--- numerator {num} ---")
            print(f"  one predictor  (persistence)        slope {b1:+.4f}   R2 {r2_1:.4f}")
            print(f"  two predictors (persistence+rms)    slopes {b1b:+.4f} / {b2b:+.4f}"
                  f"   R2 {r2_2:.4f}   incremental {r2_2 - r2_1:+.4f}")
            print(f"  residual variance left: one {1 - r2_1:.4f}, two {1 - r2_2:.4f}")
            print(f"  check corr(residual2, log persistence) {pearson(res2, lp):+.6f}, "
                  f"corr(residual2, log rms_norm) {pearson(res2, lrms):+.6f}"
                  f"   [both zero by construction]")

            print("  candidates against the two-predictor residual:")
            for c in CANDIDATES:
                r = pearson(res2, cand[c])
                p = fisher_p(r, n, 2)
                rs = spearman(res2, cand[c])
                print(f"    {c:<12} pearson {r:+.4f} (p {p:.4f})"
                      f"   spearman {rs:+.4f}{flag(p)}")

            # --- 3. log vs raw reconciliation ---
            print("  eff_rank across every normalisation tried"
                  "  (raw ratio is the shape that gives the handoff numbers):")
            e = cand["eff_rank"]
            print(f"    vs raw {num:<11} {pearson(e, v):+.4f}")
            print(f"    vs raw ratio          {pearson(e, ratio):+.4f}")
            print(f"    vs log ratio          {pearson(e, lr):+.4f}")
            print(f"    vs residual1          {pearson(e, res1):+.4f}")
            print(f"    vs residual2          {pearson(e, res2):+.4f}")

    print("\nBonferroni threshold used above: 0.0125 (four candidates).")
    print("All of this is exploratory: no statistic here was named in advance.")


if __name__ == "__main__":
    main()
