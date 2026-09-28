#!/usr/bin/env python3
"""Persistence baseline over E42 checkpoints, residual decomposition, and the
orthogonal residual construction.

v2 changes over v1:
  - writes results/persistence.json so the forward pass runs once
  - sample CV and proper median, matching the handoff's convention
  - correlations of every candidate with persistence itself
  - OLS of log(vp) on log(persistence); the residual is orthogonal to
    log(persistence) by construction, unlike the ratio
  - partial correlations of each candidate with log(vp) controlling for
    log(persistence)

Split reconstruction is verified: analyze_representation.py:47 and run_e42.py
both set NEP=200, and e39_lib.py / e41_lib.py differ only inside run_subepoch,
so collect_gym_data and DS are identical in both.

Persistence predictor: p = emb[:, 2], the last context state, against the
target tgt = emb[:, 3] that M.forward uses, weighted per item exactly as
val_loss weights.

Nothing here is pre-registered. Output is exploratory.
"""
import os
import sys
import json
import math

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(CODE)
sys.path.insert(0, CODE)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from e42_lib import FE, AE, PR, M, DS, collect_gym_data, val_loss

try:
    from e42_lib import SIGReg
    HAVE_SIG = True
except ImportError:
    HAVE_SIG = False

NEP = 200
CKPT = os.path.join(EXP, "checkpoints")
SWEEP = os.path.join(EXP, "results", "sweep.json")
PERS_OUT = os.path.join(EXP, "results", "persistence.json")

GROUPS = [("s42", 42), ("s123", 123)]
CANDIDATES = ["eff_rank", "r2_readout", "cond_number", "smoothness"]
CONTROL = ["rms_norm"]
NUMERATORS = ["best_vp", "final_vp", "mean_last3"]


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


def partial(rxy, rxz, ryz):
    d = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float("nan") if d == 0 else (rxy - rxz * ryz) / d


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def cv(v):
    """Sample CV, matching the handoff's convention."""
    m = sum(v) / len(v)
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))
    return sd / m


def spread(v):
    return max(v) / min(v)


def ols(x, y):
    """Slope, intercept, residuals of y on x."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / vx
    a0 = my - b * mx
    return b, a0, [c - (a0 + b * a) for a, c in zip(x, y)]


# ---------- split reconstruction ----------

def val_loader(data_seed):
    eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=data_seed)
    ds = DS(eps, 3)
    nt = int(len(ds) * 0.9)
    nv = len(ds) - nt
    _, va = random_split(ds, [nt, nv],
                         generator=torch.Generator().manual_seed(data_seed))
    return DataLoader(va, batch_size=64), len(ds), nt, nv, nt // 64


@torch.no_grad()
def persistence(enc, vl):
    tot, n = 0.0, 0
    for s, _a in vl:
        emb = enc(s)
        tot += F.mse_loss(emb[:, 2], emb[:, 3]).item() * s.size(0)
        n += s.size(0)
    return tot / n


@torch.no_grad()
def recompute_final_vp(seed, k, vl):
    if not HAVE_SIG:
        return None
    path = os.path.join(CKPT, f"s{seed}_i{k}_model_final.pt")
    if not os.path.exists(path):
        return None
    mdl = M(FE(), AE(), PR(), SIGReg())
    mdl.load_state_dict(torch.load(path, map_location="cpu"))
    return val_loss(mdl, vl)


# ---------- main ----------

def main():
    sd = json.load(open(SWEEP))
    cells = sd["cells"]
    print(f"cells in sweep.json: {len(cells)}   SIGReg importable: {HAVE_SIG}")

    dump = {}

    for tag, data_seed in GROUPS:
        keys = sorted((n for n in cells if n.startswith(tag + "_")),
                      key=lambda n: int(n.split("_i")[1]))
        if not keys:
            continue

        print(f"\n{'=' * 68}\n{tag}  data seed {data_seed}  n={len(keys)}\n{'=' * 68}")

        vl, n_ds, nt, nv, n_batches = val_loader(data_seed)
        stored_nb = set(cells[n]["n_batches"] for n in keys)
        print(f"split: {n_ds} windows -> train {nt}, val {nv}, n_batches "
              f"{n_batches}; stored {stored_nb}; match {stored_nb == {n_batches}}")

        rows, missing = [], []
        for n in keys:
            k = int(n.split("_i")[1])
            p = os.path.join(CKPT, f"s{data_seed}_i{k}_enc_at_freeze.pt")
            if not os.path.exists(p):
                missing.append(n)
                continue
            enc = FE()
            enc.load_state_dict(torch.load(p, map_location="cpu"))
            enc.eval()
            pv = persistence(enc, vl)
            dump[n] = pv
            rows.append((n, cells[n], pv))

        if missing:
            print(f"MISSING checkpoints: {missing}")
        if not rows:
            continue

        n0, c0, _ = rows[0]
        rec = recompute_final_vp(data_seed, int(n0.split("_i")[1]), vl)
        if rec is None:
            print("acceptance: skipped")
        else:
            st = c0["final_vp"]
            rel = abs(rec - st) / st
            print(f"acceptance {n0}: stored {st!r} recomputed {rec!r} "
                  f"rel diff {rel:.3e}"
                  + ("  WARNING: mismatch" if rel > 1e-9 else ""))

        pers = [p for _n, _c, p in rows]
        lp = [math.log(p) for p in pers]
        nn = len(rows)
        print(f"\npersistence: spread {spread(pers):.4f}x  CV {cv(pers):.4f}")

        # Is any candidate simply a persistence proxy?
        print("\ncandidates vs persistence (the proxy question):")
        cand_r = {}
        for key in CANDIDATES + CONTROL:
            v = [c[key] for _n, c, _p in rows]
            r = pearson(v, lp)
            cand_r[key] = r
            print(f"  {key:<12} vs log(persistence): pearson {r:+.4f} "
                  f"(p {fisher_p(r, nn):.4f})   spearman {spearman(v, lp):+.4f}")

        for num in NUMERATORS:
            v = [c[num] for _n, c, _p in rows]
            lv = [math.log(a) for a in v]
            ratio = [a / p for a, p in zip(v, pers)]
            lr = [math.log(a) for a in ratio]
            r_vp = pearson(lv, lp)

            print(f"\n--- numerator {num} ---")
            print(f"  raw    spread {spread(v):.4f}x  CV {cv(v):.4f}")
            print(f"  ratio  spread {spread(ratio):.4f}x  CV {cv(ratio):.4f}  "
                  f"median {median(ratio):.4f}  "
                  f"beat persistence {sum(1 for r in ratio if r < 1)}/{nn}")
            print(f"  corr log({num}) vs log(persistence): {r_vp:+.4f}")

            b, _a0, resid = ols(lp, lv)
            print(f"  OLS log({num}) on log(persistence): slope {b:+.4f} "
                  f"(division assumes 1.0000)")
            print(f"  check: corr(residual, log persistence) = "
                  f"{pearson(resid, lp):+.6f}  [zero by construction]")
            print(f"  check: corr(log ratio, log persistence) = "
                  f"{pearson(lr, lp):+.4f}  [zero only if slope is 1]")

            print("  candidate vs the two normalisations:")
            for key in CANDIDATES + CONTROL:
                xs = [c[key] for _n, c, _p in rows]
                r_ratio = pearson(lr, xs)
                r_res = pearson(resid, xs)
                r_part = partial(pearson(lv, xs), cand_r[key], r_vp)
                print(f"    {key:<12} log-ratio {r_ratio:+.4f}   "
                      f"OLS-residual {r_res:+.4f} (p {fisher_p(r_res, nn, 1):.4f})   "
                      f"partial {r_part:+.4f}")

            by_raw = sorted(range(nn), key=lambda i: v[i])
            by_ratio = sorted(range(nn), key=lambda i: ratio[i])
            by_res = sorted(range(nn), key=lambda i: resid[i])
            print(f"  best/worst  raw {rows[by_raw[0]][0]}/{rows[by_raw[-1]][0]}"
                  f"   ratio {rows[by_ratio[0]][0]}/{rows[by_ratio[-1]][0]}"
                  f"   residual {rows[by_res[0]][0]}/{rows[by_res[-1]][0]}")

    json.dump(dump, open(PERS_OUT, "w"), indent=2)
    print(f"\nwrote {PERS_OUT}  ({len(dump)} cells)")


if __name__ == "__main__":
    main()
