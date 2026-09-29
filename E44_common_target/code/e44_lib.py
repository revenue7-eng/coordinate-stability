#!/usr/bin/env python3
"""
П2 Resolution: Dim Sweep with Full Parameters
===============================================
Repeats E13 (dim sweep) with Tier 3 parameters:
- 200 episodes (was 100)
- 30 epochs (was 20)
- Predictor hidden = max(128, dim*8) (was fixed 128)

Tests dim = 1, 2, 3, 4, 5, 7, 11 on Push-T.
If prescribed_5d wins here → E13 was underpowered, П2 closed.
If prescribed_5d loses → Tier 3 result was architecture-dependent.

Run: python p2_dim_sweep_full.py
Results: p2_dim_sweep_results.json
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30
SAVE_FILE = Path("p2_dim_sweep_results.json")

# ================================================================
# Infrastructure (same as Tier 3)
# ================================================================

class SIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.linspace(0, 3, 17); dt = 3 / 16
        w = torch.full((17,), 2 * dt); w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', w * phi)
    def forward(self, proj):
        A = torch.randn(proj.size(-1), 512, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def synth(n, seed=42):
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2 * np.pi))
        st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(300):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1., 20. / dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30 - cd) / 30 * 5
                bp += (tb / cd) * f
                ba = (ba + rng.normal(0, .05) * f) % (2 * np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
            if (step + 1) % 5 == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa) + 1]), 'a': np.array(aa)})
    return eps


class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            st, a = e['s'], e['a']
            for t in range(len(a) - H):
                self.w.append((st[t:t + H + 2].astype(np.float32),
                               a[t:t + H + 1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        st, a = self.w[i]
        return torch.from_numpy(st), torch.from_numpy(a)


# ================================================================
# Encoders: prescribed and free for arbitrary dim
# ================================================================

def make_prescribed_features(x, dim):
    """Build prescribed features of given dimensionality.
    
    dim=1: [theta_norm]
    dim=2: [x_block_norm, y_block_norm]
    dim=3: [x_block_norm, y_block_norm, theta_norm]
    dim=4: [x_block, y_block, theta, sin_theta] (all normalized)
    dim=5: [x_agent, y_agent, x_block, y_block, theta] (all normalized)
    dim=7: [x_a, y_a, x_b, y_b, sin, cos, theta_norm]
    dim=11: [x_a, y_a, x_b, y_b, sin, cos, theta_norm, dx, dy, dist, x_a*x_b]
    """
    xa = x[..., 0] / 512
    ya = x[..., 1] / 512
    xb = x[..., 2] / 512
    yb = x[..., 3] / 512
    theta = x[..., 4]
    theta_norm = theta / (2 * np.pi)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    dx = xa - xb
    dy = ya - yb
    dist = torch.sqrt(dx**2 + dy**2 + 1e-8)

    if dim == 1:
        return theta_norm.unsqueeze(-1)
    elif dim == 2:
        return torch.stack([xb, yb], dim=-1)
    elif dim == 3:
        return torch.stack([xb, yb, theta_norm], dim=-1)
    elif dim == 4:
        return torch.stack([xb, yb, theta_norm, sin_t], dim=-1)
    elif dim == 5:
        return torch.stack([xa, ya, xb, yb, theta_norm], dim=-1)
    elif dim == 7:
        return torch.stack([xa, ya, xb, yb, sin_t, cos_t, theta_norm], dim=-1)
    elif dim == 11:
        return torch.stack([
            xa, ya, xb, yb, sin_t, cos_t, theta_norm,
            dx, dy, dist, xa * xb
        ], dim=-1)
    else:
        raise ValueError(f"Unsupported dim={dim}")


class PrescribedEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return make_prescribed_features(x, self.dim)


class FreeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        h = 64
        self.net = nn.Sequential(
            nn.Linear(5, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, dim))
    def forward(self, x):
        return self.net(x)


# ================================================================
# Model (same as Tier 3)
# ================================================================

class ActionEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d))
    def forward(self, a): return self.net(a)


class Predictor(nn.Module):
    def __init__(self, d, H=3):
        super().__init__()
        d_in = H * 2 * d
        h = max(128, d * 8)  # Scale with dim — KEY DIFFERENCE from E13
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, d))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(self, enc, d, H=3):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder(d)
        self.pr = Predictor(d, H)
        self.sig = SIGReg()
        self.H = H
    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :self.H], emb[:, self.H]
        aem = self.ae(a[:, :self.H])
        p = self.pr(ctx, aem)
        return {
            'pl': F.mse_loss(p, tgt.detach()),
            'sl': self.sig(emb.transpose(0, 1)),
        }


def make_data(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    eps = synth(EPISODES, seed)
    ds = SeqDS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    return tl, vl


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp = 0; n = 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item() * s.size(0); n += s.size(0)
    return tp / n


def run_condition(enc_type, dim, tl, vl, seed):
    torch.manual_seed(seed)
    if enc_type == 'prescribed':
        enc = PrescribedEncoder(dim)
    else:
        enc = FreeEncoder(dim)
    
    mdl = WorldModel(enc, dim)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        mdl.train()
        for s, a in tl:
            o = mdl(s, a)
            l = o['pl'] + 0.09 * o['sl']
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
        vl_ = val_loss(mdl, vl)
        if vl_ < best: best = vl_
    
    return best


# ================================================================
# Main
# ================================================================

def save(data):
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved to {SAVE_FILE}]")


def main():
    DIMS = [1, 2, 3, 4, 5, 7, 11]
    
    print("=" * 65)
    print("П2 RESOLUTION: DIM SWEEP WITH FULL PARAMETERS")
    print(f"Dims: {DIMS}")
    print(f"Seeds: {SEEDS}, Episodes: {EPISODES}, Epochs: {EPOCHS}")
    print(f"Predictor hidden = max(128, dim*8)")
    print("=" * 65)

    if SAVE_FILE.exists():
        with open(SAVE_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming from {SAVE_FILE}")
    else:
        all_results = {
            'config': {
                'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS,
                'dims': DIMS, 'predictor': 'max(128, dim*8)',
                'note': 'П2 resolution: same params as Tier 3'
            },
            'results': {}
        }

    total_t0 = time.time()

    for dim in DIMS:
        for seed in SEEDS:
            for enc_type in ['prescribed', 'free']:
                key = f"{enc_type}_dim{dim}_seed{seed}"
                if key in all_results['results']:
                    v = all_results['results'][key]
                    print(f"  {key}: cached ({v:.6f})")
                    continue

                t0 = time.time()
                tl, vl = make_data(seed)
                best = run_condition(enc_type, dim, tl, vl, seed)
                all_results['results'][key] = best
                dt = time.time() - t0
                print(f"  {key}: {best:.6f} ({dt:.0f}s)")
                save(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PRESCRIBED vs FREE BY DIMENSION")
    print("=" * 70)
    print(f"\n{'Dim':>5} {'Prescribed':>12} {'Free':>12} {'Ratio':>8} {'Winner':>12}")
    print("-" * 55)

    for dim in DIMS:
        p_vals = [all_results['results'].get(f'prescribed_dim{dim}_seed{s}')
                  for s in SEEDS]
        f_vals = [all_results['results'].get(f'free_dim{dim}_seed{s}')
                  for s in SEEDS]
        
        p_vals = [v for v in p_vals if v is not None]
        f_vals = [v for v in f_vals if v is not None]
        
        if p_vals and f_vals:
            pm = np.mean(p_vals)
            fm = np.mean(f_vals)
            ratio = fm / pm if pm > 0 else 0
            winner = "PRESCRIBED" if ratio > 1.2 else ("FREE" if ratio < 0.8 else "TIE")
            print(f"{dim:>5} {pm:>12.6f} {fm:>12.6f} {ratio:>7.1f}× {winner:>12}")

    # Compare with E13 (old dim sweep)
    print("\n" + "=" * 70)
    print("COMPARISON: OLD (E13, 100ep/20ep) vs NEW (200ep/30ep)")
    print("=" * 70)
    print("  Old E13 results:")
    print("    dim=3: prescribed wins 1.5×")
    print("    dim=4: FREE wins (crossover)")
    print("    dim=5: FREE wins")
    print("  New results (above) — does crossover still hold?")

    # Per-seed detail for crossover region
    print("\n--- Per-seed detail at crossover region ---")
    for dim in [3, 4, 5]:
        print(f"\n  dim={dim}:")
        for seed in SEEDS:
            pk = f'prescribed_dim{dim}_seed{seed}'
            fk = f'free_dim{dim}_seed{seed}'
            pv = all_results['results'].get(pk)
            fv = all_results['results'].get(fk)
            if pv and fv:
                ratio = fv / pv
                print(f"    seed {seed}: p={pv:.6f} f={fv:.6f} ratio={ratio:.2f}×")

    print(f"\nTotal time: {time.time()-total_t0:.0f}s")
    print(f"Results saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# E44 addition. Everything above this block is a byte-identical copy of
# E28_dim_sweep_full/code/p2_dim_sweep_full.py. This block only appends.
#
# Arms at dim 5: prescribed (PrescribedEncoder), free_raw (FreeEncoder on the
# raw state, as in E28), free_scaled (the same network on the state divided by
# its range; a buffer only, so it draws the same random stream as free_raw).
#
# Stage 1 is E28 run_condition at dim 5, unchanged (target "self").
# Stage 2 freezes the encoder and trains a fresh action encoder and predictor
# on a common target: make_prescribed_features(state at window position H, 5),
# a fixed function of the true state that does not depend on the encoder.
# For the prescribed arm this target equals its own latent, so its stage 2
# must reproduce its stage 1 bit for bit.
# ---------------------------------------------------------------------------

_E44 = {"target": "self"}


def set_target(mode):
    assert mode in ("self", "external"), mode
    _E44["target"] = mode


class FreeEncoderScaled(FreeEncoder):
    def __init__(self, dim):
        super().__init__(dim)
        self.register_buffer("sc5", torch.tensor(
            [1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, x):
        return self.net(x * self.sc5)


_WorldModel_E28 = WorldModel


class WorldModel(_WorldModel_E28):
    def forward(self, st, a):
        if _E44["target"] == "self":
            return super().forward(st, a)
        emb = self.enc(st)
        ctx = emb[:, :self.H]
        with torch.no_grad():
            tgt = make_prescribed_features(st[:, self.H], 5)
        aem = self.ae(a[:, :self.H])
        p = self.pr(ctx, aem)
        return {"pl": F.mse_loss(p, tgt),
                "sl": self.sig(emb.transpose(0, 1))}


_E44_ARMS = {"prescribed": PrescribedEncoder,
             "free_raw": FreeEncoder,
             "free_scaled": FreeEncoderScaled}


def _e44_train(mdl, tl, vl, epochs):
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    hist = []
    for ep in range(1, epochs + 1):
        mdl.train()
        for s, a in tl:
            o = mdl(s, a)
            l = o["pl"] + 0.09 * o["sl"]
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
        hist.append(val_loss(mdl, vl))
    return hist


def run_arm(arm, tl, vl, seed, stage2=True, epochs=EPOCHS):
    out = {"arm": arm, "seed": seed}
    t0 = time.time()
    torch.manual_seed(seed)
    enc = _E44_ARMS[arm](5)
    set_target("self")
    mdl = WorldModel(enc, 5)
    h1 = _e44_train(mdl, tl, vl, epochs)
    out.update(s1_hist=h1, s1_best=min(h1), s1_final=h1[-1],
               s1_seconds=round(time.time() - t0, 1))
    if not stage2:
        return out
    t0 = time.time()
    for p in enc.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    set_target("external")
    try:
        mdl2 = WorldModel(enc, 5)
        h2 = _e44_train(mdl2, tl, vl, epochs)
    finally:
        set_target("self")
    out.update(s2_hist=h2, s2_best=min(h2), s2_final=h2[-1],
               s2_seconds=round(time.time() - t0, 1))
    return out


# ---------------------------------------------------------------------------
# E44 addition, part 2, appended after acceptance. The acceptance-time file is
# a byte prefix of this one (results/acceptance_lib.sha256 holds its sha256
# and size).
#
# Arm prescribed_rotated: the prescribed features centred, multiplied by a
# fixed random orthogonal 5x5 matrix and shifted back. Same information and
# scale as prescribed, axes not aligned with the common target. The matrix is
# drawn from a private generator, so building the arm consumes nothing from
# the global random stream and stage 1 starts from the same state as for
# prescribed.
# marker: E44 rotated arm v1
# ---------------------------------------------------------------------------


class PrescribedRotated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim == 5, dim
        g = torch.Generator().manual_seed(20260928)
        q, r = torch.linalg.qr(torch.randn(5, 5, generator=g))
        q = q * torch.sign(torch.diagonal(r))
        self.register_buffer("Q", q)

    def forward(self, x):
        f = make_prescribed_features(x, 5)
        return (f - 0.5) @ self.Q + 0.5


_E44_ARMS["prescribed_rotated"] = PrescribedRotated
