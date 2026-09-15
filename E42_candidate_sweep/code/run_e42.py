#!/usr/bin/env python3
"""E42: does any cheap property of the untrained encoder predict best_vp?

Pre-registered block: EXPERIMENTS.md, E42. Predictions, candidate list,
statistics and the multiplicity correction live there and are not restated
here. {F}67 and {F}68 are the facts this experiment is sized against.

Design as registered: 30 initialisations at data seed 42, one head each, encoder
frozen at step 0, plus 8-10 initialisations at a second data seed that checks
reproduction of the spread and not any correlation.

`e42_lib.py` is a byte-identical copy of `e41_lib.py` (sha256 verified at
creation, no patch script). The candidates are computed from the
`_enc_at_freeze` checkpoint after the cell finishes, exactly as E40 computed
{F}64 and {F}67, so nothing in this experiment touches the RNG stream that
produces the encoder. `enc_seed=k, head_seed=None` is the E40 parameterisation:
initialisations 1..10 therefore reproduce E40 bit for bit and 11..30 continue
the same sweep.

Acceptance test: cell (enc_seed=1, head_seed=None) at data seed 42 reproduces
E40 init 1, best_vp = 0.0031448905217346915. If it does not, the sweep is void
and nothing here may be compared with {F}64, {F}67 or {F}68.

Usage:
  python run_e42.py probe      acceptance cell only
  python run_e42.py            seed 42 half, resumable via results/sweep.json
  python run_e42.py second     second data seed half, once SECOND_SEED is set
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE,
           "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code",
           "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e42_lib import FE, PE, collect_gym_data, run_subepoch  # noqa: E402
from analyze_representation import val_states               # noqa: E402
from e42_candidates import CANDIDATE_KEYS, CONTROL_KEYS, candidates  # noqa: E402

torch.set_num_threads(4)          # same as E40 and E41; threads move the 9th digit

BASE = "/mnt/d/coordinate-stability/E42_candidate_sweep"
OUT = os.path.join(BASE, "results")
CKPT = os.path.join(BASE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

DATA_SEED = 42
N_INITS = 30

# The pre-registration says "a second data seed" without naming one. A seed
# chosen after seeing the seed-42 results would not be pre-registered, so the
# second half refuses to run until the registry fixes the value and it is
# copied here.
SECOND_SEED = 123
N_INITS_SECOND = 10

EP, NEP = 15, 200
E40_INIT1_BEST_VP = 0.0031448905217346915

path = os.path.join(OUT, "sweep.json")
sd = json.load(open(path)) if os.path.exists(path) else {
    "epochs": EP, "episodes": NEP,
    "candidates": list(CANDIDATE_KEYS), "control": list(CONTROL_KEYS),
    "primary_seed": DATA_SEED, "n_inits": N_INITS, "cells": {}}


def save():
    json.dump(sd, open(path, "w"), indent=2)


def metrics(r):
    vp = [h["vp"] for h in r["hist"]]
    return {"best_vp": r["best_vp"],
            "final_vp": r["final_vp"],
            "mean_last3": float(np.mean(vp[-3:])),
            "argmin_ep": r["hist"][int(np.argmin(vp))]["ep"],
            "n_batches": r["n_batches"],
            "freeze_step": r["freeze_step"]}


def states_and_target(seed):
    S = val_states(seed)
    with torch.no_grad():
        target = PE()(S).numpy()
    print(f"validation states {tuple(S.shape)}", flush=True)
    return S, target


def cell(eps, seed, k, S, target):
    prefix = os.path.join(CKPT, f"s{seed}_i{k}")
    t0 = time.time()
    r = run_subepoch(eps, seed, EP, 0.0, mode="free",
                     ckpt_prefix=prefix, enc_seed=k, head_seed=None)
    enc = FE()
    enc.load_state_dict(torch.load(prefix + "_enc_at_freeze.pt",
                                   map_location="cpu"))
    enc.eval()
    row = metrics(r)
    row.update(candidates(enc, S, target))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def sweep(seed, n_inits, eps, S, target):
    head = (f"{'init':>5} {'best_vp':>11} {'eff_rank':>9} {'R2_ro':>8} "
            f"{'cond':>10} {'smooth':>8} {'rms':>8} {'s':>5}")
    print(head, flush=True)
    for k in range(1, n_inits + 1):
        key = f"s{seed}_i{k}"
        if key in sd["cells"]:
            print(f"{k:>5} skip", flush=True)
            continue
        row = cell(eps, seed, k, S, target)
        sd["cells"][key] = row
        save()
        print(f"{k:>5} {row['best_vp']:>11.6f} {row['eff_rank']:>9.4f} "
              f"{row['r2_readout']:>8.4f} {row['cond_number']:>10.3f} "
              f"{row['smoothness']:>8.4f} {row['rms_norm']:>8.4f} "
              f"{row['seconds']:>5.0f}", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else ""

if mode == "second":
    assert SECOND_SEED is not None, (
        "SECOND_SEED is unset. Name the second data seed in the E42 block of "
        "EXPERIMENTS.md and copy it here before running this half.")
    t = time.time()
    eps2 = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=SECOND_SEED)
    print(f"data collected {time.time()-t:.0f}s, {len(eps2)} episodes", flush=True)
    S2, target2 = states_and_target(SECOND_SEED)
    sd["second_seed"] = SECOND_SEED
    sd["n_inits_second"] = N_INITS_SECOND
    save()
    sweep(SECOND_SEED, N_INITS_SECOND, eps2, S2, target2)
    print(f"\ncells {len(sd['cells'])}\nwritten to {path}")
    sys.exit(0)

t = time.time()
eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=DATA_SEED)
print(f"data collected {time.time()-t:.0f}s, {len(eps)} episodes", flush=True)

t0 = time.time()
r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="free", enc_seed=1, head_seed=None)
got = r["best_vp"]
rel = abs(got - E40_INIT1_BEST_VP) / E40_INIT1_BEST_VP
print(f"\nacceptance (enc_seed=1, head_seed=None)  {time.time()-t0:.0f}s")
print(f"  expected {E40_INIT1_BEST_VP!r}")
print(f"  got      {got!r}")
print(f"  bit-exact: {got == E40_INIT1_BEST_VP}   rel diff: {rel:.3e}")
print(f"  n_batches {r['n_batches']}  freeze_step {r['freeze_step']}")
sd["acceptance"] = {"expected": E40_INIT1_BEST_VP, "got": got,
                    "bit_exact": got == E40_INIT1_BEST_VP, "rel_diff": rel}
save()
assert rel < 1e-9, f"ACCEPTANCE FAILED: rel diff {rel:.3e}, sweep is void"
print("  ACCEPTED", flush=True)

if mode == "probe":
    print("\nprobe only, stopping before the sweep")
    sys.exit(0)

S, target = states_and_target(DATA_SEED)
sweep(DATA_SEED, N_INITS, eps, S, target)

fs = {c["freeze_step"] for c in sd["cells"].values()}
nb = {c["n_batches"] for c in sd["cells"].values()}
print(f"\ncells {len(sd['cells'])}  n_batches observed {nb}  "
      f"freeze_step observed {fs}")
print(f"written to {path}")
