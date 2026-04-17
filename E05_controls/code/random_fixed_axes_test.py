#!/usr/bin/env python3
"""
Random Fixed Axes Stress Test — Push-T
========================================
Extends Experiment 2 with a critical control condition:
  - prescribed: h(s) = normalize(block_x, block_y, block_angle)
  - random_fixed: h(s) = R @ normalize(block_x, block_y, block_angle), R = random orthogonal 3x3
  - free: h(s) = learned_encoder(s)

If random_fixed ≈ prescribed → effect comes from fixation, not semantics
If random_fixed << prescribed → semantic alignment is necessary
If random_fixed > free but < prescribed → fixation helps, semantics adds

Three seeds per condition. Synthetic data mode for dependency-free reproduction.

Run:
  pip install gym-pusht torch numpy
  python random_fixed_axes_test.py
  # Or without gym-pusht:
  python random_fixed_axes_test.py --synthetic

Author: Andrew + Claude (prescribed axes project)
"""

import os, time, json, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from scipy.stats import ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# --- SIGReg (from LeWM module.py) ---
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3/(knots-1)
        w = torch.full((knots,), 2*dt); w[[0,-1]] = dt
        phi = torch.exp(-t.square()/2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w*phi)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights)*proj.size(-2)).mean()


# --- Data ---
def collect_gym_data(n_ep=200, max_steps=300, fs=5, seed=42):
    try:
        import gymnasium as gym
        import gym_pusht
    except ImportError:
        print("gym-pusht not found -> synthetic")
        return collect_synthetic_data(n_ep, max_steps, fs, seed)
    rng = np.random.default_rng(seed)
    eps = []
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    for i in range(n_ep):
        obs, _ = env.reset(seed=int(rng.integers(0,100000)))
        ss, aa = [obs.copy()], []
        for step in range(max_steps):
            if step==0 or rng.random()<0.3:
                a = env.action_space.sample()
            else:
                a = aa[-1]+rng.normal(0,30,2).astype(np.float32)
                a = np.clip(a,0,512)
            obs,_,d,tr,_ = env.step(a)
            if (step+1)%fs==0: ss.append(obs.copy()); aa.append(a.copy())
            if d or tr: break
        if len(aa)>=4:
            eps.append({"s":np.array(ss[:len(aa)+1]),"a":np.array(aa)})
        if (i+1)%50==0: print(f"  {i+1}/{n_ep}")
    env.close()
    print(f"Gym: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

def collect_synthetic_data(n_ep=200, max_steps=300, fs=5, seed=42):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_ep):
        ag = rng.uniform(50,462,2).astype(np.float32)
        bp = rng.uniform(100,412,2).astype(np.float32)
        ba = np.float32(rng.uniform(0,2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50,462,2).astype(np.float32)
        for step in range(max_steps):
            if step%20==0: tgt = rng.uniform(50,462,2).astype(np.float32)
            act = np.clip(tgt+rng.normal(0,10,2),0,512).astype(np.float32)
            d = act-ag; dn = np.linalg.norm(d)
            if dn>0: ag += d*min(1.0,20.0/dn)
            ag = np.clip(ag,0,512)
            tb = bp-ag; cd = np.linalg.norm(tb)
            if 0<cd<30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba+rng.normal(0,0.05)*f)%(2*np.pi)
            bp = np.clip(bp,0,512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
            if (step+1)%fs==0: ss.append(st.copy()); aa.append(act)
        if len(aa)>=4:
            eps.append({"s":np.array(ss[:len(aa)+1]),"a":np.array(aa)})
    print(f"Synth: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            s,a = e["s"],e["a"]
            for t in range(len(a)-H):
                self.w.append((s[t:t+H+2].astype(np.float32), a[t:t+H+1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        s,a = self.w[i]
        return torch.from_numpy(s), torch.from_numpy(a)


# --- Encoders ---
class PrescEnc(nn.Module):
    """Prescribed: extract block state and normalize."""
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, s): return s[..., 2:5] * self.sc

class RandomFixedEnc(nn.Module):
    """Random fixed: prescribed extraction + fixed random orthogonal rotation."""
    def __init__(self, rotation_seed=0):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        # Generate random orthogonal matrix (fixed, not learned)
        R = ortho_group.rvs(3, random_state=rotation_seed).astype(np.float32)
        self.register_buffer("R", torch.from_numpy(R))
        print(f"  Random rotation matrix (seed={rotation_seed}):")
        print(f"    det(R) = {np.linalg.det(R):.4f}")
        print(f"    R @ R^T ≈ I: max_err = {np.max(np.abs(R @ R.T - np.eye(3))):.2e}")
    def forward(self, s):
        normalized = s[..., 2:5] * self.sc
        return normalized @ self.R.T  # rotate in latent space

class FreeEnc(nn.Module):
    """Free: learned encoder from full state."""
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s): return self.net(s)


# --- Model ---
class ActEnc(nn.Module):
    def __init__(self, di=2, do=3, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.GELU(),nn.Linear(h,do))
    def forward(self, a): return self.net(a)

class Pred(nn.Module):
    def __init__(self, d=3, H=3, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H*2*d,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,d))
    def forward(self, e, ae):
        return self.net(torch.cat([e,ae],dim=-1).reshape(e.size(0),-1))

class Model(nn.Module):
    def __init__(self, enc, aenc, pred, sig, H=3):
        super().__init__()
        self.enc,self.aenc,self.pred,self.sig,self.H = enc,aenc,pred,sig,H
    def forward(self, s, a):
        H = self.H
        emb = self.enc(s)
        ctx, tgt = emb[:,:H], emb[:,H]
        ae = self.aenc(a[:,:H])
        p = self.pred(ctx, ae)
        return {"pl": F.mse_loss(p,tgt.detach()),
                "sl": self.sig(emb.transpose(0,1)),
                "p": p.detach(), "t": tgt.detach()}


# --- Training ---
@dataclass
class Cfg:
    n_ep:int=200; max_steps:int=300; fs:int=5
    dim:int=3; hid:int=128; H:int=3
    epochs:int=50; bs:int=128; lr:float=3e-4; wd:float=1e-3
    lam:float=0.09; split:float=0.9; out:str="random_fixed_results"

def train_ep(m, dl, opt, lam, dev):
    m.train(); tp,ts,n = 0,0,0
    for s,a in dl:
        s,a = s.to(dev),a.to(dev)
        o = m(s,a); l = o["pl"]+lam*o["sl"]
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
    return tp/n,ts/n

@torch.no_grad()
def val_ep(m, dl, dev):
    m.eval(); tp,ts,n = 0,0,0; ps,ts_ = [],[]
    for s,a in dl:
        s,a = s.to(dev),a.to(dev); o = m(s,a)
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
        ps.append(o["p"].cpu()); ts_.append(o["t"].cpu())
    p,t = torch.cat(ps),torch.cat(ts_)
    return tp/n, ts/n, (p-t).pow(2).mean(0).tolist()

def make_encoder(mode, dim, rotation_seed=0):
    if mode == "prescribed":
        return PrescEnc()
    elif mode == "random_fixed":
        return RandomFixedEnc(rotation_seed=rotation_seed)
    elif mode == "free":
        return FreeEnc(5, dim, 64)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run(mode, eps, cfg, train_seed, dev, rotation_seed=0):
    print(f"\n{'='*50}\n  {mode.upper()} (seed={train_seed})\n{'='*50}")
    ds = SeqDS(eps, cfg.H)
    nt = int(len(ds)*cfg.split); nv = len(ds)-nt
    tr,va = random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(train_seed))
    tl = DataLoader(tr,batch_size=cfg.bs,shuffle=True,drop_last=True)
    vl = DataLoader(va,batch_size=cfg.bs)
    print(f"  train={nt} val={nv}")

    enc = make_encoder(mode, cfg.dim, rotation_seed=rotation_seed)
    mdl = Model(enc, ActEnc(2,cfg.dim,32), Pred(cfg.dim,cfg.H,cfg.hid),
                SIGReg(17,512), cfg.H).to(dev)
    np_ = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  params: {np_:,}")

    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=cfg.lr,weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs)

    best,bep,hist = float("inf"),0,[]
    for ep in range(1,cfg.epochs+1):
        t0 = time.time()
        tp,ts = train_ep(mdl,tl,opt,cfg.lam,dev)
        vp,vs,axes = val_ep(mdl,vl,dev)
        sch.step()
        if vp<best: best,bep = vp,ep
        hist.append({"ep":ep,"tp":tp,"ts":ts,"vp":vp,"vs":vs,"ax":axes})
        if ep%10==0 or ep==1:
            ax = ", ".join(f"{v:.6f}" for v in axes)
            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{ax}] | {time.time()-t0:.1f}s")
    print(f"  Best: {best:.6f} (ep {bep})")
    return {"mode":mode,"seed":train_seed,"params":np_,"best":best,"bep":bep,"hist":hist}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch-size",type=int,default=128)
    pa.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    pa.add_argument("--embed-dim",type=int,default=3)
    pa.add_argument("--output-dir",type=str,default="random_fixed_results")
    pa.add_argument("--synthetic",action="store_true")
    pa.add_argument("--rotation-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for generating random orthogonal matrices (one per training seed)")
    args = pa.parse_args()

    cfg = Cfg(n_ep=args.episodes, epochs=args.epochs, bs=args.batch_size,
              dim=args.embed_dim, out=args.output_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Training seeds: {args.seeds}")
    print(f"Rotation seeds: {args.rotation_seeds}")

    # Collect data once with fixed seed
    data_seed = 42
    np.random.seed(data_seed)
    print(f"\n{'='*50}\n  DATA (seed={data_seed})\n{'='*50}")
    eps = (collect_synthetic_data if args.synthetic else collect_gym_data)(
        cfg.n_ep, cfg.max_steps, cfg.fs, data_seed)

    modes = ["prescribed", "random_fixed", "free"]
    all_results = {m: [] for m in modes}

    for i, seed in enumerate(args.seeds):
        rot_seed = args.rotation_seeds[i] if i < len(args.rotation_seeds) else i
        for mode in modes:
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = run(mode, eps, cfg, seed, dev, rotation_seed=rot_seed)
            if mode == "random_fixed":
                r["rotation_seed"] = rot_seed
            all_results[mode].append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: RANDOM FIXED AXES STRESS TEST")
    print(f"{'='*60}")

    for mode in modes:
        vals = [r["best"] for r in all_results[mode]]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {mode:15s}: {mean:.6f} ± {std:.6f}  (seeds: {[f'{v:.6f}' for v in vals]})")

    p_mean = np.mean([r["best"] for r in all_results["prescribed"]])
    rf_mean = np.mean([r["best"] for r in all_results["random_fixed"]])
    f_mean = np.mean([r["best"] for r in all_results["free"]])

    print(f"\n  Ratios:")
    print(f"    free / prescribed:        {f_mean/p_mean:.1f}×")
    print(f"    free / random_fixed:      {f_mean/rf_mean:.1f}×")
    print(f"    random_fixed / prescribed: {rf_mean/p_mean:.1f}×")

    print(f"\n  Interpretation:")
    if rf_mean / p_mean < 2.0:
        print(f"    → random_fixed ≈ prescribed: FIXATION dominates, semantics secondary")
        print(f"    → This supports gauge-fixing interpretation")
    elif rf_mean / p_mean > 5.0:
        print(f"    → random_fixed << prescribed: SEMANTIC ALIGNMENT is necessary")
        print(f"    → Fixation alone is insufficient")
    else:
        print(f"    → random_fixed between prescribed and free:")
        print(f"    → Fixation helps, semantics adds. Both factors contribute.")

    if rf_mean < f_mean:
        print(f"    → random_fixed > free: Even random fixation beats learning!")
        print(f"    → Strong evidence for gauge-fixing hypothesis")
    else:
        print(f"    → random_fixed < free: Random fixation does NOT help")
        print(f"    → Effect is specifically about semantic axes, not fixation")

    # Save
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "random_fixed_axes_stress_test",
        "description": "Tests whether prescribed axes effect comes from fixation or semantic alignment",
        "data_seed": data_seed,
        "training_seeds": args.seeds,
        "rotation_seeds": args.rotation_seeds,
        "epochs": cfg.epochs,
        "synthetic": args.synthetic,
        "results": {}
    }
    for mode in modes:
        vals = [r["best"] for r in all_results[mode]]
        summary["results"][mode] = {
            "best_per_seed": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "params": all_results[mode][0]["params"]
        }
    summary["ratios"] = {
        "free_over_prescribed": float(f_mean / p_mean),
        "free_over_random_fixed": float(f_mean / rf_mean),
        "random_fixed_over_prescribed": float(rf_mean / p_mean),
    }

    with open(out / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Save full histories
    for mode in modes:
        for r in all_results[mode]:
            fname = f"history_{mode}_seed{r['seed']}.json"
            with open(out / fname, "w") as fh:
                json.dump(r["hist"], fh)

    print(f"\n  Results saved to {out}/")


if __name__ == "__main__":
    main()
