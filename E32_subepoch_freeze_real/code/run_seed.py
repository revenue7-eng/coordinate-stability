#!/usr/bin/env python3
"""E32 per-seed runner. Usage: python run_seed.py <seed>
Reproduces the E31 sub-epoch freeze sweep on REAL gym-pusht data.
Reduced budget (epochs=4, 50 episodes) chosen to fit compute limits;
raise EP/NEP for a full-fidelity rerun."""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e32_lib import collect_gym_data, run_subepoch
import torch; torch.set_num_threads(4)
seed = int(sys.argv[1])
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(OUT, exist_ok=True)
GRID = [0.0,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,1.0]; EP, NEP = 4, 50
t = time.time(); eps = collect_gym_data(n_ep=NEP, max_steps=300, fs=5, seed=seed)
sd = {'seed': seed, 'epochs': EP, 'episodes': NEP, 'sweep': {}}
sd['prescribed'] = run_subepoch(eps, seed, EP, 0.0, mode='prescribed')['best_vp']
for f in GRID:
    sd['sweep']['%.2f'%f] = run_subepoch(eps, seed, EP, f, mode='free')['best_vp']
sd['free_unfrozen'] = run_subepoch(eps, seed, EP, 99.0, mode='free')['best_vp']
json.dump(sd, open(os.path.join(OUT, f'seed_{seed}.json'), 'w'), indent=2)
print(f'seed {seed} done {time.time()-t:.0f}s')
