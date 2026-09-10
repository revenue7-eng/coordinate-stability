#!/usr/bin/env python3
"""E38 per-seed runner. Usage: python run_seed.py <seed>
Full-fidelity rerun of E32 (EP=15, NEP=200) with sub-0.25 resolution added.
Resume-safe: re-running skips grid points already present in the seed JSON."""
import sys, json, os, time
sys.path.insert(0, "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code")
from e32_lib import collect_gym_data, run_subepoch
import torch; torch.set_num_threads(4)

seed = int(sys.argv[1])
OUT = "/mnt/d/coordinate-stability/E38_subepoch_freeze_full/results"
os.makedirs(OUT, exist_ok=True)
GRID = [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90,1.0]
EP, NEP = 15, 200
path = os.path.join(OUT, f"seed_{seed}.json")
sd = json.load(open(path)) if os.path.exists(path) else {"seed": seed, "epochs": EP, "episodes": NEP, "sweep": {}}

def save():
    json.dump(sd, open(path, "w"), indent=2)

t = time.time(); eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=seed)
print(f"data collected {time.time()-t:.0f}s", flush=True)

if "prescribed" not in sd:
    t0 = time.time(); sd["prescribed"] = run_subepoch(eps, seed, EP, 0.0, mode="prescribed")["best_vp"]
    save(); print(f"  prescribed done {time.time()-t0:.0f}s", flush=True)
for f in GRID:
    key = "%.2f" % f
    if key in sd["sweep"]:
        print(f"  f={key} skip", flush=True); continue
    t0 = time.time(); sd["sweep"][key] = run_subepoch(eps, seed, EP, f, mode="free")["best_vp"]
    save(); print(f"  f={key} done {time.time()-t0:.0f}s best_vp={sd['sweep'][key]:.5f}", flush=True)
if "free_unfrozen" not in sd:
    t0 = time.time(); sd["free_unfrozen"] = run_subepoch(eps, seed, EP, 99.0, mode="free")["best_vp"]
    save(); print(f"  free_unfrozen done {time.time()-t0:.0f}s", flush=True)
print(f"seed {seed} COMPLETE", flush=True)
