"""
E32 — sub-epoch freeze sweep on REAL gym-pusht data.
Faithful port of freeze_test_standalone.py infrastructure (prescribed-axes-drift),
extended with WITHIN-epoch-1 freezing to reproduce the E31 design on real physics.

Freeze semantics:
  freeze_frac f in [0,1] -> encoder is frozen after floor(f * n_batches_epoch1)
  optimizer steps of epoch 1, then stays frozen for all remaining epochs.
  f=0.0  == encoder frozen at init (freeze@0 proxy, ~ random_fixed / Ф12)
  f=1.0  == frozen at end of epoch 1 (freeze@1)
"""
import numpy as np, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ---- SIGReg (verbatim) ----
class SIGReg(nn.Module):
    def __init__(s, k=17, np_=512):
        super().__init__(); s.np_ = np_
        t = torch.linspace(0, 3, k); dt = 3/(k-1)
        w = torch.full((k,), 2*dt); w[[0,-1]] = dt
        phi = torch.exp(-t.square()/2.0)
        s.register_buffer('t', t); s.register_buffer('phi', phi)
        s.register_buffer('weights', w*phi)
    def forward(s, proj):
        A = torch.randn(proj.size(-1), s.np_, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * s.t
        err = (x.cos().mean(-3) - s.phi).square() + x.sin().mean(-3).square()
        return ((err @ s.weights) * proj.size(-2)).mean()

# ---- REAL data (verbatim collect_gym_data from paper2_full_analysis.py) ----
def collect_gym_data(n_ep=200, max_steps=300, fs=5, seed=42):
    import gymnasium as gym, gym_pusht
    rng = np.random.default_rng(seed); eps = []
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    for i in range(n_ep):
        obs, _ = env.reset(seed=int(rng.integers(0, 100000)))
        ss, aa = [obs.copy()], []
        for step in range(max_steps):
            if step == 0 or len(aa) == 0 or rng.random() < 0.3:
                a = env.action_space.sample()
            else:
                a = aa[-1] + rng.normal(0, 30, 2).astype(np.float32)
                a = np.clip(a, 0, 512)
            obs, _, d, tr, _ = env.step(a)
            if (step+1) % fs == 0:
                ss.append(obs.copy()); aa.append(a.copy())
            if d or tr: break
        if len(aa) >= 4:
            eps.append({"s": np.array(ss[:len(aa)+1]), "a": np.array(aa)})
    env.close()
    return eps

class DS(Dataset):
    def __init__(s, eps, H=3):
        s.w = []
        for e in eps:
            st, a = e['s'], e['a']
            for t in range(len(a)-H):
                s.w.append((st[t:t+H+2].astype(np.float32),
                            a[t:t+H+1].astype(np.float32)))
    def __len__(s): return len(s.w)
    def __getitem__(s, i):
        st, a = s.w[i]; return torch.from_numpy(st), torch.from_numpy(a)

class PE(nn.Module):  # prescribed encoder
    def __init__(s):
        super().__init__()
        s.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(s, x): return x[..., 2:5] * s.sc
class FE(nn.Module):  # free encoder
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(5,64), nn.LayerNorm(64), nn.GELU(),
                              nn.Linear(64,64), nn.LayerNorm(64), nn.GELU(),
                              nn.Linear(64,3))
    def forward(s, x): return s.net(x)
class AE(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2,32), nn.GELU(), nn.Linear(32,3))
    def forward(s, a): return s.net(a)
class PR(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(18,128), nn.LayerNorm(128), nn.GELU(),
                              nn.Linear(128,128), nn.LayerNorm(128), nn.GELU(),
                              nn.Linear(128,3))
    def forward(s, e, ae):
        return s.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))
class M(nn.Module):
    def __init__(s, enc, ae, pr, sig):
        super().__init__(); s.enc, s.ae, s.pr, s.sig = enc, ae, pr, sig
    def forward(s, st, a):
        emb = s.enc(st); ctx, tgt = emb[:, :3], emb[:, 3]
        aem = s.ae(a[:, :3]); p = s.pr(ctx, aem)
        return {'pl': F.mse_loss(p, tgt.detach()),
                'sl': s.sig(emb.transpose(0, 1))}

@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp, n = 0, 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item()*s.size(0); n += s.size(0)
    return tp/n

def run_subepoch(eps, seed, epochs, freeze_frac, mode='free'):
    """freeze_frac in [0,1]: freeze encoder after floor(frac*n_batches) steps of epoch 1."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = DS(eps, 3)
    nt = int(len(ds)*0.9); nv = len(ds)-nt
    tr, va = random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    n_batches = len(tl)
    freeze_step = int(np.floor(freeze_frac * n_batches))  # within epoch 1

    enc = PE() if mode == 'prescribed' else FE()
    mdl = M(enc, AE(), PR(), SIGReg())
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-3)
    lam = 0.09
    frozen = False
    def do_freeze():
        for p in mdl.enc.parameters(): p.requires_grad = False
        return torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                                 lr=3e-4, weight_decay=1e-3)

    # freeze BEFORE any step if frac==0
    if mode == 'free' and freeze_step == 0:
        opt = do_freeze(); frozen = True

    hist = []
    for ep in range(1, epochs+1):
        mdl.train()
        if frozen: mdl.enc.eval()
        for bi, (s, a) in enumerate(tl):
            # within-epoch-1 freeze trigger
            if mode == 'free' and (not frozen) and ep == 1 and bi == freeze_step:
                opt = do_freeze(); frozen = True; mdl.enc.eval()
            o = mdl(s, a); l = o['pl'] + lam*o['sl']
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        hist.append({'ep': ep, 'vp': val_loss(mdl, vl)})
    best = min(h['vp'] for h in hist)
    return {'freeze_frac': freeze_frac, 'best_vp': best,
            'final_vp': hist[-1]['vp'], 'n_batches': n_batches,
            'freeze_step': freeze_step, 'hist': hist}
