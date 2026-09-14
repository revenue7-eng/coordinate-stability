#!/usr/bin/env python3
"""E41: does the spread measured in E40 belong to the encoder at all?

E40 varied one `init_seed` that drew FE, AE and PR from the same stream, and the
encoder was frozen at step 0, so only AE+PR ever trained. The 7.31x spread in
best_vp is therefore a spread over initialisations of the whole model, and Г27
asks to explain it with a property of the encoder. That attribution is untested.

Design: crossed grid, 8 encoder seeds x 5 head seeds, one fixed data sample
(seed 42). Encoder seeds 1..8 are bit-identical to E40 inits 1..8, because the
stream position before FE() is unchanged, so their R2_readout and eff_rank are
already in E40_init_sweep/results/sweep.json and are not recomputed here.

Two-way crossed random effects, no replication within a cell (a run is
deterministic given the seed pair). df 7 / 4 / 28; interaction is confounded
with the residual. Reported quantity: the share of variance attributable to the
encoder factor.

PREDICTION REGISTERED BEFORE THE RUN (2026-09-14, Claude):
  - share of best_vp variance attributable to the encoder is below 0.5,
    point estimate about 0.3; refuted if it is 0.5 or above.
  - the encoder share on mean_last3 is higher than on best_vp, because best_vp
    is a minimum over epochs and inflates spread by itself.

Acceptance test: cell (enc_seed=1, head_seed=None) reproduces E40 init 1,
best_vp = 0.0031448905217346915. If it does not, the patch moved the stream and
the whole grid is void.

Usage:
  python run_e41.py probe     acceptance cell only
  python run_e41.py           full grid (resumable via results/grid.json)
"""
import sys, os, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e41_lib import collect_gym_data, run_subepoch

torch.set_num_threads(4)          # same as E40; threads move the 9th digit

BASE = "/mnt/d/coordinate-stability/E41_variance_decomp"
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

DATA_SEED = 42
EP, NEP = 15, 200
ENC_SEEDS = list(range(1, 9))
HEAD_SEEDS = [101, 102, 103, 104, 105]
E40_INIT1_BEST_VP = 0.0031448905217346915

probe_only = len(sys.argv) > 1 and sys.argv[1] == "probe"
path = os.path.join(OUT, "grid.json")
gd = json.load(open(path)) if os.path.exists(path) else {
    "data_seed": DATA_SEED, "epochs": EP, "episodes": NEP,
    "enc_seeds": ENC_SEEDS, "head_seeds": HEAD_SEEDS, "cells": {}}

def save():
    json.dump(gd, open(path, "w"), indent=2)

def metrics(r):
    vp = [h["vp"] for h in r["hist"]]
    amin = int(np.argmin(vp))
    return {"best_vp": r["best_vp"],
            "final_vp": r["final_vp"],
            "mean_last3": float(np.mean(vp[-3:])),
            "argmin_ep": r["hist"][amin]["ep"],
            "n_batches": r["n_batches"],
            "freeze_step": r["freeze_step"]}

t = time.time()
eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=DATA_SEED)
print(f"data collected {time.time()-t:.0f}s, {len(eps)} episodes", flush=True)

# ---- acceptance ----
t0 = time.time()
r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="free", enc_seed=1, head_seed=None)
got = r["best_vp"]
rel = abs(got - E40_INIT1_BEST_VP) / E40_INIT1_BEST_VP
print(f"\nacceptance (enc_seed=1, head_seed=None)  {time.time()-t0:.0f}s")
print(f"  expected {E40_INIT1_BEST_VP!r}")
print(f"  got      {got!r}")
print(f"  bit-exact: {got == E40_INIT1_BEST_VP}   rel diff: {rel:.3e}")
print(f"  n_batches {r['n_batches']}  freeze_step {r['freeze_step']}")
gd["acceptance"] = {"expected": E40_INIT1_BEST_VP, "got": got,
                    "bit_exact": got == E40_INIT1_BEST_VP, "rel_diff": rel}
save()
assert rel < 1e-9, f"ACCEPTANCE FAILED: rel diff {rel:.3e}, grid is void"
print("  ACCEPTED", flush=True)

if probe_only:
    print("\nprobe only, stopping before the grid")
    sys.exit(0)

# ---- grid ----
print(f"\n{'enc':>4} {'head':>5} {'best_vp':>11} {'final_vp':>11} "
      f"{'mean_l3':>11} {'argmin':>7} {'s':>5}")
for e in ENC_SEEDS:
    for h in HEAD_SEEDS:
        key = f"e{e}_h{h}"
        if key in gd["cells"]:
            print(f"{e:>4} {h:>5} skip", flush=True); continue
        t0 = time.time()
        r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="free",
                         enc_seed=e, head_seed=h)
        m = metrics(r)
        gd["cells"][key] = m
        save()
        print(f"{e:>4} {h:>5} {m['best_vp']:>11.6f} {m['final_vp']:>11.6f} "
              f"{m['mean_last3']:>11.6f} {m['argmin_ep']:>7} "
              f"{time.time()-t0:>5.0f}", flush=True)

nb = {c["n_batches"] for c in gd["cells"].values()}
fs = {c["freeze_step"] for c in gd["cells"].values()}
print(f"\ncells {len(gd['cells'])}/{len(ENC_SEEDS)*len(HEAD_SEEDS)}  "
      f"n_batches observed {nb}  freeze_step observed {fs}")
print(f"written to {path}")
