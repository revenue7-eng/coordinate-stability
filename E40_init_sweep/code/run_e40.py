#!/usr/bin/env python3
"""E40: does the result depend on what the initialisation carries, or only on
the fact that it stops moving?

Г26 says immobility, not informativeness. The five points available from E39
order best_vp inversely to the initialisation's R2_readout, but there
initialisation and data sample vary together, so they cannot separate the two.

Here the data seed is fixed and only the encoder initialisation varies. The
encoder is frozen at step 0, so it never trains: best_vp then measures how good
that fixed initialisation is as a coordinate system, and R2_readout and eff_rank
measure what it carries.

Prediction under Г26: best_vp is roughly flat across initialisations and does
not track R2_readout. If best_vp tracks R2_readout instead, Г26 is refuted and
informativeness is what matters.

A prescribed run is included as the reference point (R2_readout = 1 by
construction).

Usage: python run_e40.py [n_inits]     (default 10)
"""
import sys, os, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code")
sys.path.insert(0, "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code")
from e40_lib import collect_gym_data, run_subepoch, FE, PE
from analyze_representation import r2_readout, eff_rank, val_states

torch.set_num_threads(4)

BASE = "/mnt/d/coordinate-stability/E40_init_sweep"
OUT = os.path.join(BASE, "results")
CKPT = os.path.join(BASE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

DATA_SEED = 42
EP, NEP = 15, 200

n_inits = int(sys.argv[1]) if len(sys.argv) > 1 else 10
INITS = list(range(1, n_inits + 1))

path = os.path.join(OUT, "sweep.json")
sd = json.load(open(path)) if os.path.exists(path) else {
    "data_seed": DATA_SEED, "epochs": EP, "episodes": NEP, "inits": {}}

def save():
    json.dump(sd, open(path, "w"), indent=2)

t = time.time()
eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=DATA_SEED)
print(f"data collected {time.time()-t:.0f}s", flush=True)

S = val_states(DATA_SEED)
with torch.no_grad():
    target = PE()(S).numpy()
print(f"validation states {tuple(S.shape)}", flush=True)

if "prescribed" not in sd:
    t0 = time.time()
    r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="prescribed")
    sd["prescribed"] = {"best_vp": r["best_vp"], "r2_readout": 1.0}
    save()
    print(f"  prescribed done {time.time()-t0:.0f}s "
          f"best_vp={r['best_vp']:.5f}", flush=True)

print(f"{'init':>5} {'best_vp':>10} {'R2_readout':>11} {'eff_rank':>9}")
for k in INITS:
    key = str(k)
    if key in sd["inits"]:
        print(f"{k:>5} skip", flush=True); continue
    prefix = os.path.join(CKPT, f"init{k}")
    t0 = time.time()
    r = run_subepoch(eps, DATA_SEED, EP, 0.0, mode="free",
                     ckpt_prefix=prefix, init_seed=k)
    enc = FE()
    enc.load_state_dict(torch.load(prefix + "_enc_at_freeze.pt",
                                   map_location="cpu"))
    enc.eval()
    with torch.no_grad():
        rep = enc(S).numpy()
    row = {"best_vp": r["best_vp"],
           "r2_readout": r2_readout(rep, target),
           "eff_rank": eff_rank(rep),
           "freeze_step": r.get("freeze_step")}
    sd["inits"][key] = row
    save()
    print(f"{k:>5} {row['best_vp']:>10.5f} {row['r2_readout']:>11.4f} "
          f"{row['eff_rank']:>9.4f}   ({time.time()-t0:.0f}s)", flush=True)

rows = [sd["inits"][str(k)] for k in INITS if str(k) in sd["inits"]]
if len(rows) >= 3:
    bv = np.array([r["best_vp"] for r in rows])
    r2 = np.array([r["r2_readout"] for r in rows])
    er = np.array([r["eff_rank"] for r in rows])
    print(f"\nbest_vp    min {bv.min():.5f}  max {bv.max():.5f}  "
          f"spread {bv.max()/bv.min():.2f}x")
    print(f"R2_readout min {r2.min():.4f}  max {r2.max():.4f}")
    print(f"corr(best_vp, R2_readout) = {np.corrcoef(bv, r2)[0,1]:+.3f}")
    print(f"corr(best_vp, eff_rank)   = {np.corrcoef(bv, er)[0,1]:+.3f}")
    print(f"prescribed best_vp        = {sd['prescribed']['best_vp']:.5f}")
print(f"\nwritten to {path}")
