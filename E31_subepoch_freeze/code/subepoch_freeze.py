#!/usr/bin/env python3
"""
Experiment 31: Sub-Epoch Freeze Sweep
======================================
E30 localized the free-encoder damage to the FIRST epoch (freeze@0->@1 = 136x cliff).
This resolves "within epoch 1" into a specific point by freezing the encoder at
fractions of the first epoch: after 0%, 25%, 50%, 75%, 100% of epoch-1 batches.

If a sharp fraction f* exists where best_vp jumps, that is the window boundary.
If the decline is smooth across f, the damage is continuous within epoch 1 (no sharp point).

DIRECT EXTENSION of exp7_freeze/freeze_test_standalone.py — identical infrastructure
(synth, DS, PE/FE/AE/PR, SIGReg, lr=3e-4, lambda=0.09, batch=64). The ONLY change:
freeze is triggered mid-epoch by batch fraction, not at an epoch boundary.

Run:   python subepoch_freeze.py [--episodes N --epochs E --seeds ...]
Output: results/subepoch_freeze_results.json

Note: freeze@0.0 = freeze before any gradient step = random-init frozen encoder
(the freeze@0 / random_fixed condition E30 used as proxy — here it is LITERAL).
"""
import numpy as np, json, time, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

torch.set_num_threads(1)

# === Infrastructure (identical to exp7) ===
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

def synth(n, seed=42):
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(300):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1., 20./dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba + rng.normal(0, .05)*f) % (2*np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
            if (step+1) % 5 == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa)+1]), 'a': np.array(aa)})
    return eps

class DS(Dataset):
    def __init__(s, eps, H=3):
        s.w = []
        for e in eps:
            st, a = e['s'], e['a']
            for t in range(len(a)-H):
                s.w.append((st[t:t+H+2].astype(np.float32), a[t:t+H+1].astype(np.float32)))
    def __len__(s): return len(s.w)
    def __getitem__(s, i):
        st, a = s.w[i]; return torch.from_numpy(st), torch.from_numpy(a)

class PE(nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer('sc', torch.tensor([1/512,1/512,1/(2*np.pi)]))
    def forward(s, x): return x[..., 2:5] * s.sc
class FE(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(5,64),nn.LayerNorm(64),nn.GELU(),
                              nn.Linear(64,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,3))
    def forward(s, x): return s.net(x)
class AE(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,3))
    def forward(s, a): return s.net(a)
class PR(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(18,128),nn.LayerNorm(128),nn.GELU(),
                              nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,3))
    def forward(s, e, ae): return s.net(torch.cat([e,ae],-1).reshape(e.size(0),-1))
class M(nn.Module):
    def __init__(s, enc, ae, pr, sig):
        super().__init__(); s.enc,s.ae,s.pr,s.sig = enc,ae,pr,sig
    def forward(s, st, a):
        emb = s.enc(st); ctx,tgt = emb[:,:3],emb[:,3]; aem = s.ae(a[:,:3]); p = s.pr(ctx,aem)
        return {'pl':F.mse_loss(p,tgt.detach()),'sl':s.sig(emb.transpose(0,1))}

@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp,n = 0,0
    for s,a in vl:
        o = mdl(s,a); tp += o['pl'].item()*s.size(0); n += s.size(0)
    return tp/n

def freeze_encoder(mdl):
    for p in mdl.enc.parameters(): p.requires_grad = False
    return torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-3)

def run_subepoch(frac, eps, seed, epochs):
    """frac in [0,1]: fraction of FIRST epoch's batches after which encoder is frozen.
       frac=0.0 -> freeze before first step (literal freeze@0). frac=1.0 -> freeze after epoch 1."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = DS(eps, 3)
    nt = int(len(ds)*0.9); nv = len(ds)-nt
    tr, va = random_split(ds, [nt,nv], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    n_batches = len(tl)
    freeze_after = int(round(frac * n_batches))  # batch index after which to freeze

    mdl = M(PE() if False else FE(), AE(), PR(), SIGReg())  # always free encoder here
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-3)

    frozen = False
    if freeze_after == 0:  # freeze before any step
        opt = freeze_encoder(mdl); frozen = True

    hist = []
    for ep in range(1, epochs+1):
        mdl.train()
        if frozen: mdl.enc.eval()
        for bi,(s,a) in enumerate(tl):
            # mid-epoch freeze trigger (only during epoch 1)
            if (not frozen) and ep == 1 and bi >= freeze_after:
                opt = freeze_encoder(mdl); frozen = True; mdl.enc.eval()
            o = mdl(s,a); l = o['pl'] + 0.09*o['sl']
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(),1.0); opt.step()
        hist.append({'ep':ep,'vp':val_loss(mdl,vl)})
    best = min(h['vp'] for h in hist)
    return {'frac':frac,'freeze_after_batch':freeze_after,'n_batches':n_batches,
            'best_vp':best,'final_vp':hist[-1]['vp'],'hist':hist}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42,123,777,7,99])
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0])
    ap.add_argument('--out', default='results/subepoch_freeze_results.json')
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = {'config':{'episodes':args.episodes,'epochs':args.epochs,'seeds':args.seeds,
                         'fracs':args.fracs,'note':'sub-epoch freeze within epoch 1; extends exp7'},
               'by_frac':{}, 'partial':{}}

    # ---- RESUME: if output file already exists, load it and skip finished work ----
    if Path(args.out).exists():
        try:
            prev = json.load(open(args.out))
            results['by_frac'] = prev.get('by_frac', {})
            results['partial'] = prev.get('partial', {})
            done = list(results['by_frac'].keys())
            if done:
                print(f"[resume] found existing {args.out}; already-complete fracs: {done}", flush=True)
        except Exception as e:
            print(f"[resume] could not read {args.out} ({e}); starting fresh", flush=True)

    for frac in args.fracs:
        fk = str(frac)
        if fk in results['by_frac']:
            print(f"[skip] frac={frac:.2f} already complete", flush=True)
            continue
        # recover any per-seed results saved before an interruption within this frac
        per_seed_map = dict(results['partial'].get(fk, {}))  # {seed_str: best_vp}
        for seed in args.seeds:
            if str(seed) in per_seed_map:
                print(f"[skip] frac={frac:.2f} seed={seed} already done", flush=True)
                continue
            t0 = time.time()
            eps = synth(args.episodes, seed)
            r = run_subepoch(frac, eps, seed, args.epochs)
            per_seed_map[str(seed)] = float(r['best_vp'])
            print(f"frac={frac:.2f} seed={seed}: best_vp={r['best_vp']:.6f} "
                  f"(freeze after batch {r['freeze_after_batch']}/{r['n_batches']}, {time.time()-t0:.0f}s)", flush=True)
            # SAVE after every seed so an interruption loses at most one run
            results['partial'][fk] = per_seed_map
            json.dump(results, open(args.out,'w'), indent=2, ensure_ascii=False)
        # frac complete: aggregate and move from partial -> by_frac
        arr = np.array([per_seed_map[str(s)] for s in args.seeds])
        results['by_frac'][fk] = {'mean':float(arr.mean()),'std':float(arr.std()),
                                  'per_seed':[float(x) for x in arr]}
        results['partial'].pop(fk, None)
        json.dump(results, open(args.out,'w'), indent=2, ensure_ascii=False)

    # readout (only over fracs that are complete)
    fracs = [f for f in args.fracs if str(f) in results['by_frac']]
    means = [results['by_frac'][str(f)]['mean'] for f in fracs]
    print("\n=== SUB-EPOCH PROFILE ===")
    for f,m in zip(fracs,means): print(f"  freeze@{f:.2f} epoch1: best_vp={m:.6f}")
    if len(means) < 2:
        print("(need >=2 complete fracs for profile analysis)")
        json.dump(results, open(args.out,'w'), indent=2, ensure_ascii=False)
        print(f"\nsaved {args.out}"); return
    diffs = np.diff(means)
    print(f"consecutive diffs: {np.round(diffs,6)}")
    rng = max(means)-min(means)
    if rng>0:
        big = int(np.argmax(np.abs(diffs)))
        print(f"largest jump: frac {fracs[big]}->{fracs[big+1]} = {diffs[big]:.6f} "
              f"({abs(diffs[big])/rng:.0%} of range)")
        print("=> sharp sub-epoch boundary" if abs(diffs[big])/rng>0.6 else "=> smooth within epoch 1")
    json.dump(results, open(args.out,'w'), indent=2, ensure_ascii=False)
    print(f"\nsaved {args.out}")

if __name__ == '__main__':
    main()
