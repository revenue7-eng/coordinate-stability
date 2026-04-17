#!/usr/bin/env python3
"""
Noise Control Experiment (E29)
==============================
Goal: Distinguish structured drift from generic noise.

Design: Take prescribed encoder (fixed, works well).
Add Gaussian noise to its output with sigma matched to
measured drift magnitudes from free encoder (E06).

If noise degrades as much as drift → drift = just noise.
If noise degrades LESS → drift destroys something specific
(coordinate identity across time) that random noise does not.

Key insight: drift is CORRELATED across time (the coordinate
system rotates systematically), while noise is i.i.d.
A predictor can learn to ignore i.i.d. noise (it averages out
over the context window H=3), but cannot track systematic
coordinate rotation.

Conditions:
1. prescribed (baseline)
2. prescribed + constant noise (σ matched to LATE drift ~0.005)
3. prescribed + constant noise (σ matched to MID drift ~0.08)
4. prescribed + constant noise (σ matched to EARLY drift ~1.4)
5. prescribed + epoch-varying noise (σ follows actual drift schedule)
6. prescribed + CORRELATED noise (same noise vector per sample within epoch,
   changes between epochs — mimics drift structure)
7. free encoder (reference)

3 seeds, 30 epochs, 200 episodes synthetic.

Run: python noise_control.py
Results: noise_control_results.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30
SAVE_FILE = Path("../results/noise_control_results.json")

# Drift schedule from E06 (mean across 3 seeds)
# raw_drift per epoch transition
DRIFT_SCHEDULE = [
    1.4283, 0.5792, 0.1187, 0.0926, 0.0795,
    0.1029, 0.0816, 0.0772, 0.1195, 0.1457,
    0.0564, 0.0429, 0.0376, 0.0439, 0.0410,
    0.0291, 0.0298, 0.0188, 0.0165, 0.0177,
    0.0151, 0.0123, 0.0134, 0.0086, 0.0100,
    0.0065, 0.0060, 0.0056, 0.0045, 0.0012,
]

def drift_to_sigma(drift_magnitude, dim=3):
    """Convert mean L2 drift to per-dimension Gaussian sigma."""
    from scipy.special import gamma as gamma_fn
    expected_norm_per_sigma = np.sqrt(2) * gamma_fn((dim+1)/2) / gamma_fn(dim/2)
    return drift_magnitude / expected_norm_per_sigma


# ================================================================
# Infrastructure (from tier2)
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
# Encoders
# ================================================================

class PrescribedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, x): return x[..., 2:5] * self.sc


class NoisyPrescribedEncoder(nn.Module):
    """Prescribed encoder + i.i.d. Gaussian noise per sample."""
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        self.sigma = sigma  # can be updated per epoch

    def forward(self, x):
        clean = x[..., 2:5] * self.sc
        if self.training:
            noise = torch.randn_like(clean) * self.sigma
            return clean + noise
        else:
            return clean  # no noise at eval


class CorrelatedNoisePrescribedEncoder(nn.Module):
    """Prescribed encoder + noise that is CONSTANT within an epoch
    but CHANGES between epochs. This mimics drift structure:
    coordinate system shifts systematically, not randomly per sample.
    
    Implementation: at the start of each epoch, sample a random
    displacement vector. Add it to all embeddings during that epoch.
    Next epoch: new displacement vector.
    """
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        self.sigma = sigma
        self.register_buffer('displacement', torch.zeros(3))

    def new_epoch_displacement(self):
        """Call at the start of each training epoch."""
        self.displacement = torch.randn(3, device=self.sc.device) * self.sigma

    def forward(self, x):
        clean = x[..., 2:5] * self.sc
        if self.training:
            return clean + self.displacement
        else:
            return clean


class FreeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)


# ================================================================
# Model
# ================================================================

class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))
    def forward(self, a): return self.net(a)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 3))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(self, enc, use_sigreg=True):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder()
        self.pr = Predictor()
        self.sig = SIGReg() if use_sigreg else None

    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :3], emb[:, 3]
        aem = self.ae(a[:, :3])
        p = self.pr(ctx, aem)
        result = {'pl': F.mse_loss(p, tgt.detach()), 'emb': emb.detach()}
        if self.sig is not None:
            result['sl'] = self.sig(emb.transpose(0, 1))
        else:
            result['sl'] = torch.tensor(0.0)
        return result


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


def train_epoch(mdl, tl, opt, lam=0.09):
    mdl.train()
    for s, a in tl:
        o = mdl(s, a)
        l = o['pl'] + lam * o['sl']
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()


# ================================================================
# Experiment
# ================================================================

def run_condition(label, enc_fn, seed, use_sigreg=True, epoch_callback=None):
    """Run one condition, return best_val_loss and per-epoch history."""
    tl, vl = make_data(seed)
    torch.manual_seed(seed)
    enc = enc_fn()
    mdl = WorldModel(enc, use_sigreg=use_sigreg)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)

    best = float('inf')
    history = []
    for ep in range(1, EPOCHS + 1):
        if epoch_callback is not None:
            epoch_callback(mdl, ep)
        train_epoch(mdl, tl, opt, lam=0.09 if use_sigreg else 0.0)
        vl_ = val_loss(mdl, vl)
        if vl_ < best:
            best = vl_
        history.append({'ep': ep, 'vp': vl_})

    return best, history


def run_seed(seed):
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    results = {}

    # 1. Prescribed (clean baseline)
    print("  [1/8] prescribed (clean)")
    best, hist = run_condition("prescribed", PrescribedEncoder, seed)
    results['prescribed'] = {'best': best, 'history': hist}
    print(f"        best = {best:.6f}")

    # 2. Prescribed + late noise (σ matched to epoch 28-29 drift ≈ 0.005)
    sigma_late = drift_to_sigma(0.005)
    print(f"  [2/8] prescribed + noise_late (σ={sigma_late:.4f}, drift≈0.005)")
    best, hist = run_condition("noise_late",
        lambda: NoisyPrescribedEncoder(sigma_late), seed)
    results['noise_late'] = {'best': best, 'sigma': sigma_late, 'drift': 0.005, 'history': hist}
    print(f"        best = {best:.6f}")

    # 3. Prescribed + mid noise (σ matched to epoch 2-3 drift ≈ 0.08)
    sigma_mid = drift_to_sigma(0.08)
    print(f"  [3/8] prescribed + noise_mid (σ={sigma_mid:.4f}, drift≈0.08)")
    best, hist = run_condition("noise_mid",
        lambda: NoisyPrescribedEncoder(sigma_mid), seed)
    results['noise_mid'] = {'best': best, 'sigma': sigma_mid, 'drift': 0.08, 'history': hist}
    print(f"        best = {best:.6f}")

    # 4. Prescribed + early noise (σ matched to epoch 0-1 drift ≈ 1.43)
    sigma_early = drift_to_sigma(1.43)
    print(f"  [4/8] prescribed + noise_early (σ={sigma_early:.4f}, drift≈1.43)")
    best, hist = run_condition("noise_early",
        lambda: NoisyPrescribedEncoder(sigma_early), seed)
    results['noise_early'] = {'best': best, 'sigma': sigma_early, 'drift': 1.43, 'history': hist}
    print(f"        best = {best:.6f}")

    # 5. Prescribed + epoch-varying noise (follows drift schedule)
    print(f"  [5/8] prescribed + noise_schedule (epoch-varying)")
    def schedule_callback(mdl, ep):
        if ep <= len(DRIFT_SCHEDULE):
            mdl.enc.sigma = drift_to_sigma(DRIFT_SCHEDULE[ep - 1])
    best, hist = run_condition("noise_schedule",
        lambda: NoisyPrescribedEncoder(drift_to_sigma(DRIFT_SCHEDULE[0])),
        seed, epoch_callback=schedule_callback)
    results['noise_schedule'] = {'best': best, 'history': hist}
    print(f"        best = {best:.6f}")

    # 6. Correlated noise (constant within epoch, changes between epochs)
    # Use mid-level sigma to see the effect clearly
    sigma_corr = drift_to_sigma(0.08)
    print(f"  [6/8] prescribed + correlated_noise (σ={sigma_corr:.4f})")
    def corr_callback(mdl, ep):
        mdl.enc.new_epoch_displacement()
    best, hist = run_condition("correlated_mid",
        lambda: CorrelatedNoisePrescribedEncoder(sigma_corr),
        seed, epoch_callback=corr_callback)
    results['correlated_mid'] = {'best': best, 'sigma': sigma_corr, 'history': hist}
    print(f"        best = {best:.6f}")

    # 7. Correlated noise with drift schedule
    sigma_corr_early = drift_to_sigma(1.43)
    print(f"  [7/8] prescribed + correlated_schedule (epoch-varying)")
    def corr_schedule_callback(mdl, ep):
        if ep <= len(DRIFT_SCHEDULE):
            mdl.enc.sigma = drift_to_sigma(DRIFT_SCHEDULE[ep - 1])
        mdl.enc.new_epoch_displacement()
    best, hist = run_condition("correlated_schedule",
        lambda: CorrelatedNoisePrescribedEncoder(drift_to_sigma(DRIFT_SCHEDULE[0])),
        seed, epoch_callback=corr_schedule_callback)
    results['correlated_schedule'] = {'best': best, 'history': hist}
    print(f"        best = {best:.6f}")

    # 8. Free encoder (reference)
    print("  [8/8] free encoder")
    best, hist = run_condition("free", FreeEncoder, seed)
    results['free'] = {'best': best, 'history': hist}
    print(f"        best = {best:.6f}")

    return results


def main():
    print("=" * 60)
    print("  Noise Control Experiment (E29)")
    print("  prescribed + noise vs free encoder")
    print("=" * 60)

    all_results = {'config': {
        'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS,
        'drift_schedule': DRIFT_SCHEDULE,
        'note': 'noise sigma matched to free encoder drift magnitude from E06'
    }}

    for seed in SEEDS:
        all_results[f'seed_{seed}'] = run_seed(seed)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY (mean across seeds)")
    print("=" * 60)

    conditions = ['prescribed', 'noise_late', 'noise_mid', 'noise_early',
                   'noise_schedule', 'correlated_mid', 'correlated_schedule', 'free']

    for cond in conditions:
        vals = [all_results[f'seed_{s}'][cond]['best'] for s in SEEDS]
        mean = np.mean(vals)
        std = np.std(vals)
        ratio_vs_prescribed = mean / np.mean([
            all_results[f'seed_{s}']['prescribed']['best'] for s in SEEDS])
        ratio_vs_free = mean / np.mean([
            all_results[f'seed_{s}']['free']['best'] for s in SEEDS])
        print(f"  {cond:<25} mean={mean:.6f} ±{std:.6f}  "
              f"vs_prescribed={ratio_vs_prescribed:8.2f}×  "
              f"vs_free={ratio_vs_free:.4f}×")

    # Save
    with open(SAVE_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
