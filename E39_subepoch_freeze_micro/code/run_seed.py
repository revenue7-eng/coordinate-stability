#!/usr/bin/env python3
"""E39 per-seed runner. Micro-grid inside the first 8 optimizer steps of epoch 1.

Resolves the window left unmeasured by E38, whose finest step (f=0.05) is 8
batches at n_batches=160. Tests whether a collapse-then-recover transient hides
below E38's resolution: such a transient requires a non-monotone dip, and a
frozen encoder cannot recover from one.

Grid is specified in optimizer steps, not fractions. f = (step + 0.5)/n_batches
so that floor(f * n_batches) lands on the intended step regardless of float
representation. Steps 0 and 8 use E38's literal f values (0.00, 0.05) and must
reproduce E38's numbers for the same seeds.

Data collection comes from e39_lib, which seeds the action space; e32_lib does
not, so its collect_gym_data does not reproduce across processes for a given seed.

Stores the full run_subepoch return plus two checkpoints per point.
Resume-safe: re-running skips grid points already present in the seed JSON.

Usage: python run_seed.py <seed>
"""
import sys, json, os, time

sys.path.insert(0, "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e39_lib import collect_gym_data, run_subepoch
import torch; torch.set_num_threads(4)

seed = int(sys.argv[1])
BASE = "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro"
OUT = os.path.join(BASE, "results")
CKPT = os.path.join(BASE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

N_BATCHES = 160          # measured for NEP=200, fs=5, batch_size=64, drop_last
EP, NEP = 15, 200
STEPS = [0, 1, 2, 3, 4, 6, 8]

def frac_for(step):
    if step == 0:
        return 0.0                    # E38 anchor, literal
    if step == 8:
        return 0.05                   # E38 anchor, literal
    return (step + 0.5) / N_BATCHES

path = os.path.join(OUT, f"seed_{seed}.json")
sd = json.load(open(path)) if os.path.exists(path) else {
    "seed": seed, "epochs": EP, "episodes": NEP,
    "n_batches_expected": N_BATCHES, "sweep": {}}

def save():
    json.dump(sd, open(path, "w"), indent=2)

t = time.time(); eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=seed)
print(f"data collected {time.time()-t:.0f}s", flush=True)

for step in STEPS:
    key = "%d" % step
    if key in sd["sweep"]:
        print(f"  step={step} skip", flush=True); continue
    f = frac_for(step)
    prefix = os.path.join(CKPT, f"seed{seed}_step{step}")
    t0 = time.time()
    r = run_subepoch(eps, seed, EP, f, mode="free", ckpt_prefix=prefix)
    r["f_requested"] = f
    r["ckpt_prefix"] = prefix
    sd["sweep"][key] = r
    save()
    got = r.get("freeze_step")
    flag = "" if got == step else f"  MISMATCH expected {step}"
    print(f"  step={step} f={f:.6f} done {time.time()-t0:.0f}s "
          f"best_vp={r['best_vp']:.5f} freeze_step={got}{flag}", flush=True)

nb = {v.get("n_batches") for v in sd["sweep"].values()}
print(f"seed {seed} COMPLETE  n_batches observed: {nb}", flush=True)
