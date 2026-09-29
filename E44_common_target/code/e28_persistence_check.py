# Ф77 check on E28's own synthetic dynamics, no training and no free encoder:
# how often the block moves, and the one-step persistence MSE of each
# prescribed latent, next to the E28 losses of both encoders.
import sys, json, numpy as np, torch
sys.path.insert(0, "E28_dim_sweep_full/code")
import p2_dim_sweep_full as P
R = json.load(open("E28_dim_sweep_full/results/p2_dim_sweep_results.json"))["results"]
for s in P.SEEDS:
    eps = P.synth(P.EPISODES, s)
    moved = np.mean(np.concatenate(
        [np.abs(np.diff(e["s"][:, 2:5], axis=0)).sum(1) > 1e-6 for e in eps]))
    print(f"seed {s}: episodes {len(eps)}, fraction of steps where the block moved {moved:.4f}")
    for d in [1, 2, 3, 4, 5, 7, 11]:
        sq = []
        for e in eps:
            f = P.make_prescribed_features(torch.from_numpy(e["s"]), d)
            sq.append(((f[1:] - f[:-1]) ** 2).mean(-1))
        pers = torch.cat(sq).mean().item()
        p = R.get(f"prescribed_dim{d}_seed{s}"); fr = R.get(f"free_dim{d}_seed{s}")
        print(f"  dim {d:2d}: persistence {pers:.6f}  E28 prescribed {p:.6f}  E28 free {fr:.6f}  ratio {fr / p:8.1f}")
