#!/usr/bin/env python3
"""
Two Critical Control Experiments — Push-T
==========================================
Experiment A: Gauge-Fixed Free Encoder
  - Free MLP encoder, but after each optimizer step, embeddings are
    Procrustes-aligned to the previous epoch's embedding space.
  - If quality improves vs vanilla free → drift causes quality loss (causation proved)
  - If no improvement → drift is symptom, not cause

Experiment B: Linear Free Encoder
  - Learnable linear projection (5→3), no nonlinearity
  - Same function class as prescribed (linear), but weights are learned
  - Isolates fixation vs learning at equal expressiveness
  - If linear_free ≈ prescribed → expressiveness confound eliminated
  - If linear_free ≈ free → nonlinearity is not the issue

Also runs prescribed and vanilla free for comparison (4 conditions total).

Run:
  py -3.12 gauge_fix_experiments.py --synthetic --epochs 50 --seeds 42 123 777

Author: Andrew + Claude (prescribed axes project)
"""

import os, time, json, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from scipy.linalg import orthogonal_procrustes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# --- SIGReg ---
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
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, s): return s[..., 2:5] * self.sc

class FreeEnc(nn.Module):
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s): return self.net(s)

class LinearFreeEnc(nn.Module):
    """Linear free encoder: learnable linear projection 5→3, no nonlinearity."""
    def __init__(self, di=5, do=3):
        super().__init__()
        self.linear = nn.Linear(di, do, bias=True)
    def forward(self, s): return self.linear(s)


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
                "p": p.detach(), "t": tgt.detach(),
                "emb": emb.detach()}


# --- Gauge fixing logic ---
@torch.no_grad()
def collect_embeddings(model, dataloader, dev):
    """Collect all embeddings from encoder for Procrustes alignment."""
    model.eval()
    all_emb = []
    for s, a in dataloader:
        s = s.to(dev)
        emb = model.enc(s)[:, 0]  # first timestep
        all_emb.append(emb.cpu())
    return torch.cat(all_emb).numpy()

@torch.no_grad()
def apply_gauge_fix(model, R_fix, bias_fix):
    """Apply rotation to the last layer of the encoder to align with reference frame."""
    # For FreeEnc: last layer is net[-1] which is Linear(h, do)
    last_layer = model.enc.net[-1]
    W = last_layer.weight.data.clone()  # [do, h]
    b = last_layer.bias.data.clone()    # [do]

    # Apply rotation: new_output = R @ old_output + bias_shift
    # Which means: new_W = R @ W, new_b = R @ b + bias_shift
    R_torch = torch.from_numpy(R_fix.astype(np.float32)).to(W.device)
    b_torch = torch.from_numpy(bias_fix.astype(np.float32)).to(W.device)

    last_layer.weight.data = R_torch @ W
    last_layer.bias.data = R_torch @ b + b_torch


# --- Training ---
@dataclass
class Cfg:
    n_ep:int=200; max_steps:int=300; fs:int=5
    dim:int=3; hid:int=128; H:int=3
    epochs:int=50; bs:int=128; lr:float=3e-4; wd:float=1e-3
    lam:float=0.09; split:float=0.9; out:str="gauge_fix_results"

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

def make_encoder(mode, dim):
    if mode == "prescribed":
        return PrescEnc()
    elif mode == "free":
        return FreeEnc(5, dim, 64)
    elif mode == "linear_free":
        return LinearFreeEnc(5, dim)
    elif mode == "gauge_fixed_free":
        return FreeEnc(5, dim, 64)  # same architecture, gauge fixing applied during training
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run(mode, eps, cfg, train_seed, dev, out_dir):
    print(f"\n{'='*50}\n  {mode.upper()} (seed={train_seed})\n{'='*50}")
    ds = SeqDS(eps, cfg.H)
    nt = int(len(ds)*cfg.split); nv = len(ds)-nt
    tr,va = random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(train_seed))
    tl = DataLoader(tr,batch_size=cfg.bs,shuffle=True,drop_last=True)
    vl = DataLoader(va,batch_size=cfg.bs)
    # Full dataset loader for gauge fixing (no shuffle, no drop_last)
    full_dl = DataLoader(ds, batch_size=cfg.bs, shuffle=False)
    print(f"  train={nt} val={nv}")

    enc = make_encoder(mode, cfg.dim)
    mdl = Model(enc, ActEnc(2,cfg.dim,32), Pred(cfg.dim,cfg.H,cfg.hid),
                SIGReg(17,512), cfg.H).to(dev)
    np_ = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  params: {np_:,}")

    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=cfg.lr,weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs)

    best,bep,hist = float("inf"),0,[]
    prev_emb = None  # for gauge fixing
    gauge_fix_count = 0

    for ep in range(1,cfg.epochs+1):
        t0 = time.time()
        tp,ts = train_ep(mdl,tl,opt,cfg.lam,dev)

        # Gauge fixing: after each epoch, align current embeddings to previous
        if mode == "gauge_fixed_free" and prev_emb is not None:
            curr_emb = collect_embeddings(mdl, full_dl, dev)

            # Center both
            prev_centered = prev_emb - prev_emb.mean(0)
            curr_centered = curr_emb - curr_emb.mean(0)

            # Find optimal rotation: curr → prev
            R, _ = orthogonal_procrustes(curr_centered, prev_centered)

            # Bias adjustment
            bias_fix = prev_emb.mean(0) - (curr_emb.mean(0) @ R)

            # Apply rotation to encoder's last layer
            apply_gauge_fix(mdl, R, bias_fix)
            gauge_fix_count += 1

        # Collect embeddings for next epoch's gauge fixing
        if mode == "gauge_fixed_free":
            prev_emb = collect_embeddings(mdl, full_dl, dev)

        vp,vs,axes = val_ep(mdl,vl,dev)
        sch.step()
        if vp<best: best,bep = vp,ep
        hist.append({"ep":ep,"tp":tp,"ts":ts,"vp":vp,"vs":vs,"ax":axes})
        if ep%10==0 or ep==1:
            ax = ", ".join(f"{v:.6f}" for v in axes)
            extra = f" | fixes={gauge_fix_count}" if mode == "gauge_fixed_free" else ""
            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{ax}] | {time.time()-t0:.1f}s{extra}")

    print(f"  Best: {best:.6f} (ep {bep})")

    # Save
    prefix = f"{mode}_seed{train_seed}"
    torch.save(mdl.state_dict(), out_dir / f"{prefix}_model.pt")
    with open(out_dir / f"{prefix}_history.json", "w") as fh:
        json.dump(hist, fh)

    return {"mode": mode, "seed": train_seed, "params": np_, "best": best, "bep": bep, "hist": hist}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch-size",type=int,default=128)
    pa.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    pa.add_argument("--embed-dim",type=int,default=3)
    pa.add_argument("--output-dir",type=str,default="gauge_fix_results")
    pa.add_argument("--synthetic",action="store_true")
    args = pa.parse_args()

    cfg = Cfg(n_ep=args.episodes, epochs=args.epochs, bs=args.batch_size,
              dim=args.embed_dim, out=args.output_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Seeds: {args.seeds}")

    data_seed = 42
    np.random.seed(data_seed)
    print(f"\n{'='*50}\n  DATA (seed={data_seed})\n{'='*50}")
    eps = (collect_synthetic_data if args.synthetic else collect_gym_data)(
        cfg.n_ep, cfg.max_steps, cfg.fs, data_seed)

    modes = ["prescribed", "free", "gauge_fixed_free", "linear_free"]
    all_results = {m: [] for m in modes}

    for seed in args.seeds:
        for mode in modes:
            torch.manual_seed(seed)
            np.random.seed(seed)
            r = run(mode, eps, cfg, seed, dev, out_dir)
            all_results[mode].append(r)

    # ==========================================
    #  SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: GAUGE FIXING & LINEAR FREE EXPERIMENTS")
    print(f"{'='*60}")

    for mode in modes:
        vals = [r["best"] for r in all_results[mode]]
        mean = np.mean(vals)
        std = np.std(vals)
        params = all_results[mode][0]["params"]
        print(f"  {mode:20s}: {mean:.6f} ± {std:.6f}  (params: {params:,})")

    p_mean = np.mean([r["best"] for r in all_results["prescribed"]])
    f_mean = np.mean([r["best"] for r in all_results["free"]])
    gf_mean = np.mean([r["best"] for r in all_results["gauge_fixed_free"]])
    lf_mean = np.mean([r["best"] for r in all_results["linear_free"]])

    print(f"\n  Ratios (vs prescribed):")
    print(f"    free / prescribed:              {f_mean/p_mean:.1f}×")
    print(f"    gauge_fixed_free / prescribed:   {gf_mean/p_mean:.1f}×")
    print(f"    linear_free / prescribed:        {lf_mean/p_mean:.1f}×")

    print(f"\n  Ratios (key comparisons):")
    print(f"    gauge_fixed_free / free:         {gf_mean/f_mean:.3f}× {'(BETTER)' if gf_mean < f_mean else '(WORSE)'}")
    print(f"    linear_free / free:              {lf_mean/f_mean:.3f}× {'(BETTER)' if lf_mean < f_mean else '(WORSE)'}")

    # Interpretation
    print(f"\n  === EXPERIMENT A: GAUGE-FIXED FREE ===")
    improvement = (f_mean - gf_mean) / f_mean * 100
    if gf_mean < f_mean * 0.5:
        print(f"    Gauge fixing improved free by {improvement:.1f}%")
        print(f"    → DRIFT IS CAUSAL: removing drift improves quality")
        print(f"    → Causation established, not just correlation")
    elif gf_mean < f_mean * 0.9:
        print(f"    Gauge fixing improved free by {improvement:.1f}%")
        print(f"    → Drift contributes to quality loss, but is not the sole cause")
    else:
        print(f"    Gauge fixing did not significantly improve free ({improvement:.1f}%)")
        print(f"    → Drift may be a symptom, not a cause")
        print(f"    → Or: Procrustes alignment interferes with optimization")

    print(f"\n  === EXPERIMENT B: LINEAR FREE ===")
    if lf_mean / p_mean < 2.0:
        print(f"    linear_free ≈ prescribed (ratio {lf_mean/p_mean:.2f}×)")
        print(f"    → Expressiveness confound ELIMINATED")
        print(f"    → Effect is about fixation, not about nonlinearity")
    elif lf_mean / p_mean < 5.0:
        print(f"    linear_free between prescribed and free (ratio {lf_mean/p_mean:.2f}×)")
        print(f"    → Partial expressiveness confound")
        print(f"    → Linear learning captures some structure but drifts")
    else:
        print(f"    linear_free ≈ free (ratio {lf_mean/p_mean:.1f}×)")
        print(f"    → Expressiveness is NOT the confound")
        print(f"    → Learning itself (even linear) causes drift")

    # Save
    summary = {
        "experiment": "gauge_fixing_and_linear_free",
        "data_seed": data_seed,
        "training_seeds": args.seeds,
        "epochs": cfg.epochs,
        "synthetic": args.synthetic,
        "results": {},
        "ratios": {}
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
        "gauge_fixed_over_prescribed": float(gf_mean / p_mean),
        "linear_free_over_prescribed": float(lf_mean / p_mean),
        "gauge_fixed_over_free": float(gf_mean / f_mean),
        "linear_free_over_free": float(lf_mean / f_mean),
    }

    with open(out_dir / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
