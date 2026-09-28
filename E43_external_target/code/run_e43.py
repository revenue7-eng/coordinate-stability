#!/usr/bin/env python3
"""E43: the E42 sweep with an external prediction target.

Pre-registered block: EXPERIMENTS.md, E43. Hypotheses, candidates, the named
statistic and the multiplicity correction live there and are not restated here.

Design: the encoders of E42 (enc_seed=k, head_seed=None, so initialisation k at
a data seed is the same frozen encoder as E42 cell s<seed>_i<k>), each with one
head trained to predict PE(state at window position 3) instead of the encoder's
own output there. Everything else is the E42 cell.

e43_lib.py is e42_lib.py byte for byte plus an appended block that rebinds M.

Checks:
  1. Acceptance, target "self", before any external cell: enc_seed=1 at data
     seed 42 reproduces E40 init 1, best_vp = 0.0031448905217346915.
  2. Every external cell: the frozen encoder saved at freeze equals the E42
     checkpoint of the same cell tensor for tensor. The pre-registered
     candidates and persistence are then copied from E42 rather than
     recomputed, since they are functions of that encoder alone.

The external sweep refuses to run until PREREG_COMMIT holds the hash of the
commit that adds the E43 block to EXPERIMENTS.md.

Usage:
  python run_e43.py probe      acceptance only, target "self"
  python run_e43.py            data seed 42, external target, resumable
  python run_e43.py second     data seed 123, external target, resumable
  python run_e43.py floor      encoder 1 at data seed 42, head_seed 1..5,
                               external target, resumable
"""
import json
import os
import subprocess
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

from e43_lib import FE, collect_gym_data, run_subepoch, set_target, get_target  # noqa: E402

torch.set_num_threads(4)          # same as E40, E41, E42; threads move the 9th digit

ROOT = "/mnt/d/coordinate-stability"
BASE = os.path.join(ROOT, "E43_external_target")
OUT = os.path.join(BASE, "results")
CKPT = os.path.join(BASE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

E42_BASE = os.path.join(ROOT, "E42_candidate_sweep")
E42_CELLS = json.load(open(os.path.join(E42_BASE, "results", "sweep.json")))["cells"]
E42_PERS = json.load(open(os.path.join(E42_BASE, "results", "persistence.json")))
E42_CKPT = os.path.join(E42_BASE, "checkpoints")
COPY_KEYS = ["eff_rank", "r2_readout", "cond_number", "smoothness", "rms_norm"]

# Hash of the commit that adds the E43 block to EXPERIMENTS.md. The external
# sweep asserts it is set and is an ancestor of HEAD.
PREREG_COMMIT = None

DATA_SEED = 42
N_INITS = 30
SECOND_SEED = 123
N_INITS_SECOND = 10

EP, NEP = 15, 200
E40_INIT1_BEST_VP = 0.0031448905217346915

path = os.path.join(OUT, "sweep.json")
sd = json.load(open(path)) if os.path.exists(path) else {
    "target": "external", "epochs": EP, "episodes": NEP,
    "primary_seed": DATA_SEED, "n_inits": N_INITS, "cells": {}}


def save():
    json.dump(sd, open(path, "w"), indent=2)


def git_head():
    return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                   text=True).strip()


def require_prereg():
    assert PREREG_COMMIT is not None, (
        "PREREG_COMMIT is unset. Commit the E43 block of EXPERIMENTS.md first, "
        "then copy that commit hash here.")
    rc = subprocess.call(["git", "-C", ROOT, "merge-base", "--is-ancestor",
                          PREREG_COMMIT, "HEAD"])
    assert rc == 0, f"PREREG_COMMIT {PREREG_COMMIT} is not an ancestor of HEAD"
    reg = subprocess.check_output(["git", "-C", ROOT, "show",
                                   f"{PREREG_COMMIT}:EXPERIMENTS.md"], text=True)
    assert reg.count("\n### E43.") == 1, (
        f"PREREG_COMMIT {PREREG_COMMIT} does not contain the E43 block of "
        "EXPERIMENTS.md, so it is not the pre-registration commit")
    files = subprocess.check_output(["git", "-C", ROOT, "show", "--name-only",
                                     "--format=", PREREG_COMMIT], text=True)
    assert "E43_external_target/code/analyze_e43.py" in files, (
        f"PREREG_COMMIT {PREREG_COMMIT} does not add analyze_e43.py")
    sd["prereg_commit"] = PREREG_COMMIT
    sd.setdefault("heads_at_run", []).append(git_head())
    save()


def metrics(r):
    vp = [h["vp"] for h in r["hist"]]
    return {"best_vp": r["best_vp"],
            "final_vp": r["final_vp"],
            "mean_last3": float(np.mean(vp[-3:])),
            "argmin_ep": r["hist"][int(np.argmin(vp))]["ep"],
            "n_batches": r["n_batches"],
            "freeze_step": r["freeze_step"]}


def same_encoder(prefix, seed, k):
    ours = torch.load(prefix + "_enc_at_freeze.pt", map_location="cpu")
    ref = torch.load(os.path.join(E42_CKPT, f"s{seed}_i{k}_enc_at_freeze.pt"),
                     map_location="cpu")
    if set(ours) != set(ref):
        return False
    return all(torch.equal(ours[n], ref[n]) for n in ours)


def cell(eps, seed, k, head_seed=None):
    assert get_target() == "external"
    key = f"s{seed}_i{k}" if head_seed is None else f"s{seed}_i{k}_h{head_seed}"
    prefix = os.path.join(CKPT, key)
    t0 = time.time()
    r = run_subepoch(eps, seed, EP, 0.0, mode="free",
                     ckpt_prefix=prefix, enc_seed=k, head_seed=head_seed)
    assert same_encoder(prefix, seed, k), (
        f"{key}: frozen encoder differs from the E42 checkpoint, cell is void")
    ref = E42_CELLS[f"s{seed}_i{k}"]
    row = metrics(r)
    row["encoder_matches_e42"] = True
    for c in COPY_KEYS:
        row[c] = ref[c]
    row["persistence"] = E42_PERS[key]
    row["e42_best_vp"] = ref["best_vp"]
    row["e42_final_vp"] = ref["final_vp"]
    row["head_seed"] = head_seed
    row["seconds"] = round(time.time() - t0, 1)
    return row


def sweep(seed, n_inits, eps):
    print(f"{'init':>5} {'final_vp':>11} {'best_vp':>11} {'e42_final':>11} "
          f"{'persist':>9} {'R2_ro':>7} {'eff_rank':>8} {'s':>5}", flush=True)
    for k in range(1, n_inits + 1):
        key = f"s{seed}_i{k}"
        if key in sd["cells"]:
            print(f"{k:>5} skip", flush=True)
            continue
        row = cell(eps, seed, k)
        sd["cells"][key] = row
        save()
        print(f"{k:>5} {row['final_vp']:>11.6f} {row['best_vp']:>11.6f} "
              f"{row['e42_final_vp']:>11.6f} {row['persistence']:>9.6f} "
              f"{row['r2_readout']:>7.4f} {row['eff_rank']:>8.4f} "
              f"{row['seconds']:>5.0f}", flush=True)


def acceptance(eps):
    set_target("self")
    t0 = time.time()
    r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="free", enc_seed=1, head_seed=None)
    got = r["best_vp"]
    rel = abs(got - E40_INIT1_BEST_VP) / E40_INIT1_BEST_VP
    print(f"\nacceptance, target self (enc_seed=1, head_seed=None)  {time.time()-t0:.0f}s")
    print(f"  expected {E40_INIT1_BEST_VP!r}")
    print(f"  got      {got!r}")
    print(f"  bit-exact: {got == E40_INIT1_BEST_VP}   rel diff: {rel:.3e}")
    print(f"  n_batches {r['n_batches']}  freeze_step {r['freeze_step']}")
    sd["acceptance"] = {"target": "self", "expected": E40_INIT1_BEST_VP, "got": got,
                        "bit_exact": got == E40_INIT1_BEST_VP, "rel_diff": rel}
    save()
    assert rel < 1e-9, f"ACCEPTANCE FAILED: rel diff {rel:.3e}, E43 is void"
    print("  ACCEPTED", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else ""

if mode == "second":
    require_prereg()
    t = time.time()
    eps2 = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=SECOND_SEED)
    print(f"data collected {time.time()-t:.0f}s, {len(eps2)} episodes", flush=True)
    set_target("external")
    sd["second_seed"] = SECOND_SEED
    sd["n_inits_second"] = N_INITS_SECOND
    save()
    sweep(SECOND_SEED, N_INITS_SECOND, eps2)
    print(f"\ncells {len(sd['cells'])}\nwritten to {path}")
    sys.exit(0)

t = time.time()
eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=DATA_SEED)
print(f"data collected {time.time()-t:.0f}s, {len(eps)} episodes", flush=True)

if mode == "floor":
    require_prereg()
    set_target("external")
    print(f"{'head':>5} {'final_vp':>11} {'best_vp':>11} {'s':>5}", flush=True)
    for j in range(1, 6):
        key = f"s{DATA_SEED}_i1_h{j}"
        if key in sd["cells"]:
            print(f"{j:>5} skip", flush=True)
            continue
        row = cell(eps, DATA_SEED, 1, head_seed=j)
        sd["cells"][key] = row
        save()
        print(f"{j:>5} {row['final_vp']:>11.6f} {row['best_vp']:>11.6f} "
              f"{row['seconds']:>5.0f}", flush=True)
    print(f"written to {path}")
    sys.exit(0)

acceptance(eps)

if mode == "probe":
    print("\nprobe only, target self, stopping before any external cell")
    sys.exit(0)

require_prereg()
set_target("external")
sweep(DATA_SEED, N_INITS, eps)

fs = {c["freeze_step"] for c in sd["cells"].values()}  # floor cells included
nb = {c["n_batches"] for c in sd["cells"].values()}
print(f"\ncells {len(sd['cells'])}  n_batches observed {nb}  "
      f"freeze_step observed {fs}")
print(f"written to {path}")
