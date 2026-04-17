"""
Prescribed Axes Dimension Sweep: Double Pendulum
==================================================
Internal dimensionality = 4 (θ1, θ2, ω1, ω2)

State: [θ1, ω1, θ2, ω2]
Full state available to both prescribed and free.

Prescribed axes cumulative:
  1: θ1
  2: + θ2
  3: + ω1
  4: + ω2
  5: + sinθ1 (redundant)
  6: + sinθ2 (redundant)
  7: + cosθ1 (redundant)
  8: + cosθ2 (redundant)

Usage: python run_double_pendulum.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path


def generate_double_pendulum_episodes(n_episodes=100, steps=50, seed=0):
    """
    Double pendulum with Euler integration.
    Simplified equations (equal masses, equal lengths = 1).
    State: [θ1, ω1, θ2, ω2]
    """
    rng = np.random.RandomState(seed)
    g = 9.81
    dt = 0.01
    substeps = 5  # 5 substeps per recorded step
    episodes = []

    for _ in range(n_episodes):
        th1 = rng.uniform(-np.pi/2, np.pi/2)
        th2 = rng.uniform(-np.pi/2, np.pi/2)
        w1 = rng.uniform(-1, 1)
        w2 = rng.uniform(-1, 1)

        states = []
        for t in range(steps):
            states.append([th1, w1, th2, w2])
            for _ in range(substeps):
                delta = th2 - th1
                den1 = 3 - np.cos(2 * delta)
                den2 = den1  # simplified

                a1 = (-g * (2 * np.sin(th1) - np.sin(th1 - 2*th2))
                       - 2 * np.sin(delta) * (w2**2 + w1**2 * np.cos(delta))) / den1
                a2 = (2 * np.sin(delta) * (2 * w1**2 + 2 * g * np.cos(th1)
                       - w2**2 * np.cos(delta))) / den2

                w1 += a1 * dt
                w2 += a2 * dt
                th1 += w1 * dt
                th2 += w2 * dt

                # Damping
                w1 *= 0.999
                w2 *= 0.999

                # Wrap angles
                th1 = ((th1 + np.pi) % (2 * np.pi)) - np.pi
                th2 = ((th2 + np.pi) % (2 * np.pi)) - np.pi

        episodes.append(np.array(states, dtype=np.float32))
    return episodes


def compute_prescribed(states_t, dim):
    """
    Prescribed axes for double pendulum.
      1: θ1 / π
      2: θ2 / π
      3: ω1 / 5
      4: ω2 / 5
      5: sin(θ1)
      6: sin(θ2)
      7: cos(θ1)
      8: cos(θ2)
    """
    th1 = states_t[:, 0]
    w1 = states_t[:, 1]
    th2 = states_t[:, 2]
    w2 = states_t[:, 3]

    all_axes = [
        th1 / np.pi,       # 0
        th2 / np.pi,       # 1
        w1 / 5.0,          # 2
        w2 / 5.0,          # 3
        np.sin(th1),       # 4
        np.sin(th2),       # 5
        np.cos(th1),       # 6
        np.cos(th2),       # 7
    ]

    return np.stack(all_axes[:dim], axis=-1).astype(np.float32)


def make_pairs_prescribed(episodes, dim):
    contexts, targets = [], []
    for ep in episodes:
        for t in range(len(ep) - 1):
            ctx = compute_prescribed(ep[t:t+1], dim)
            tgt = compute_prescribed(ep[t+1:t+2], dim)
            contexts.append(ctx[0])
            targets.append(tgt[0])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


def make_pairs_raw(episodes):
    """Raw 4D state normalized."""
    contexts, targets = [], []
    for ep in episodes:
        normed = ep.copy()
        normed[:, 0] /= np.pi
        normed[:, 1] /= 5.0
        normed[:, 2] /= np.pi
        normed[:, 3] /= 5.0
        for t in range(len(ep) - 1):
            contexts.append(normed[t])
            targets.append(normed[t + 1])
    return (torch.tensor(np.array(contexts), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32))


class Predictor(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, dim))
    def forward(self, x): return self.net(x)


class FreeEncoder(nn.Module):
    def __init__(self, in_d=4, out_d=4, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)


def train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, free_enc=None,
          epochs=20, bs=128, lr=3e-4):
    pred = Predictor(dim)
    params = list(pred.parameters())
    if free_enc: params += list(free_enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = float("inf")
    n = len(ctx_tr)

    for ep in range(epochs):
        pred.train()
        if free_enc: free_enc.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            c, t = ctx_tr[idx], tgt_tr[idx]
            if free_enc: c, t = free_enc(c), free_enc(t)
            loss = nn.functional.mse_loss(pred(c), t.detach() if free_enc else t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sch.step()
        pred.eval()
        if free_enc: free_enc.eval()
        with torch.no_grad():
            c, t = ctx_va, tgt_va
            if free_enc: c, t = free_enc(c), free_enc(t)
            vl = nn.functional.mse_loss(pred(c), t).item()
        if vl < best: best = vl
    return best


def main():
    print("Double Pendulum — Dimension Sweep")
    print("Internal dimensionality = 4 (θ1, θ2, ω1, ω2)")
    print("=" * 55)

    episodes = generate_double_pendulum_episodes(100)
    n_tr = 80
    dims = [1, 2, 3, 4, 5, 6, 7, 8]
    seeds = [42, 123, 777]

    results = {}
    print(f"\n{'Dim':>4} {'Prescribed':>12} {'Free':>12} {'Ratio':>8} {'Winner':>12}")
    print("-" * 55)

    crossover = None
    for dim in dims:
        p_vals, f_vals = [], []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            ctx_tr, tgt_tr = make_pairs_prescribed(episodes[:n_tr], dim)
            ctx_va, tgt_va = make_pairs_prescribed(episodes[n_tr:], dim)
            p = train(ctx_tr, tgt_tr, ctx_va, tgt_va, dim, epochs=20)
            p_vals.append(p)

            torch.manual_seed(seed); np.random.seed(seed)
            cr, tr = make_pairs_raw(episodes[:n_tr])
            cv, tv = make_pairs_raw(episodes[n_tr:])
            fe = FreeEncoder(4, dim)
            f = train(cr, tr, cv, tv, dim, free_enc=fe, epochs=20)
            f_vals.append(f)

        pm, fm = np.mean(p_vals), np.mean(f_vals)
        ratio = fm / pm if pm > 0 else 0
        winner = "PRESCRIBED" if pm < fm else "FREE"
        print(f"{dim:>4} {pm:>12.6f} {fm:>12.6f} {ratio:>7.1f}× {winner:>12}")

        results[dim] = {"prescribed": float(pm), "free": float(fm),
                        "ratio": float(ratio), "winner": winner}

        if pm > fm and crossover is None:
            crossover = dim

    print(f"\n{'='*55}")
    if crossover:
        print(f"Crossover at dim={crossover}")
        print(f"Predicted internal dimensionality: {crossover - 1}")
        print(f"Actual internal dimensionality: 4")
        print(f"Match: {'YES' if crossover - 1 == 4 else 'NO'}")
    else:
        print("No crossover found — checking if prescribed ever wins")
        any_prescribed = any(v["winner"] == "PRESCRIBED" for v in results.values())
        if not any_prescribed:
            print("Prescribed never wins on double pendulum")
        else:
            print("Prescribed wins at all tested dimensions")

    out = Path("results_double_pendulum")
    out.mkdir(exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}/results.json")


if __name__ == "__main__":
    main()
