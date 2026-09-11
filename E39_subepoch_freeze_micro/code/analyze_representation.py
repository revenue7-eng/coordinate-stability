#!/usr/bin/env python3
"""Г26 probe: what changes in the representation as the freeze step moves.

Reads the E39 encoder checkpoints (encoder state at the moment of freezing) and,
for each grid point, measures three things on the same validation states:

  R2_readout   R^2 of a linear map from the representation to the true
               coordinates (block x, y, angle), scaled as the prescribed
               encoder scales them. This is the discriminating quantity.
               If it holds steady while best_vp grows, the information is
               preserved and only the basis moves. If it falls, information
               is being lost and the claim has to change.

  procrustes   Disparity between this point's representation and the previous
               point's, after optimal centering, scaling and rotation. How far
               the basis travelled per optimizer step, at constant content.

  eff_rank     exp of the entropy of the normalised covariance spectrum,
               between 1 and 3 here. Control against collapse.

The encoder output is 3-dimensional, so subspace angles and rank carry little
information; the linear readout is what separates the hypotheses.

Reference point: the prescribed encoder is x[...,2:5]*sc, a parameter-free
readout of the true state, so its R2_readout is 1 by construction. It is
printed as a sanity check on the pipeline, not as a result.

Usage: python analyze_representation.py [seed ...]     (default: all five)
"""
import sys, os, json
import numpy as np
import torch
from torch.utils.data import random_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code")
from e39_lib import collect_gym_data, DS, FE, PE

BASE = "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro"
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

STEPS = [0, 1, 2, 3, 4, 6, 8]
EP, NEP = 15, 200
torch.set_num_threads(4)


def val_states(seed):
    """Validation states, split exactly as run_subepoch splits them."""
    eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=seed)
    ds = DS(eps, 3)
    nt = int(len(ds) * 0.9)
    nv = len(ds) - nt
    _, va = random_split(ds, [nt, nv],
                         generator=torch.Generator().manual_seed(seed))
    S = torch.stack([va[i][0] for i in range(len(va))])   # (Nv, T, 5)
    return S.reshape(-1, S.shape[-1])                     # (N, 5)


def r2_readout(rep, target):
    """R^2 of the least-squares linear map rep -> target, pooled over dims."""
    X = np.concatenate([rep, np.ones((rep.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    pred = X @ coef
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def procrustes(A, B):
    """Disparity between A and B after centering, scaling, optimal rotation."""
    A = A - A.mean(0); B = B - B.mean(0)
    nA = np.linalg.norm(A); nB = np.linalg.norm(B)
    if nA == 0 or nB == 0:
        return float("nan")
    A = A / nA; B = B / nB
    u, s, vt = np.linalg.svd(A.T @ B)
    return float(max(0.0, 1.0 - s.sum() ** 2))


def eff_rank(rep):
    c = np.cov(rep.T)
    w = np.linalg.eigvalsh(c)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return float("nan")
    p = w / w.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def run(seed):
    S = val_states(seed)
    with torch.no_grad():
        pe = PE()
        target = pe(S).numpy()                    # true coords, scaled
    sweep = json.load(open(os.path.join(RES, f"seed_{seed}.json")))["sweep"]

    print(f"\nseed {seed}   N={S.shape[0]}   "
          f"prescribed R2_readout={r2_readout(target, target):.4f} (sanity)")
    print(f"{'step':>4} {'best_vp':>10} {'R2_readout':>11} "
          f"{'procrustes':>11} {'eff_rank':>9}")

    rows, prev = [], None
    for step in STEPS:
        path = os.path.join(CKPT, f"seed{seed}_step{step}_enc_at_freeze.pt")
        enc = FE()
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        enc.eval()
        with torch.no_grad():
            rep = enc(S).numpy()
        r2 = r2_readout(rep, target)
        pr = procrustes(prev, rep) if prev is not None else float("nan")
        er = eff_rank(rep)
        bv = sweep[str(step)]["best_vp"]
        rows.append({"step": step, "best_vp": bv, "r2_readout": r2,
                     "procrustes_from_prev": pr, "eff_rank": er})
        print(f"{step:>4} {bv:>10.5f} {r2:>11.4f} {pr:>11.5f} {er:>9.4f}")
        prev = rep

    json.dump(rows, open(os.path.join(OUT, f"repr_seed_{seed}.json"), "w"),
              indent=2)
    return rows


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 123, 777, 2024, 7]
    for s in seeds:
        run(s)
    print(f"\nwritten to {OUT}")
