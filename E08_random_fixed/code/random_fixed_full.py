#!/usr/bin/env python3
"""
Random Fixed Axes + Procrustes Alignment Test — Push-T
========================================================
Full version: trains all conditions, saves ALL intermediate data,
then runs Procrustes alignment analysis.

Saves per condition per seed:
  - model weights (.pt)
  - embeddings at epochs 1, 10, 25, 50 (.npz)
  - training history (.json)
  - covariance matrices at each saved epoch (.npz)

Then runs:
  - Procrustes alignment: free→prescribed, free→random_fixed
  - R² analysis
  - Eigenvalue stability analysis

Run:
  py -3.12 random_fixed_full.py --synthetic --epochs 50 --seeds 42 123 777

Author: Andrew + Claude (prescribed axes project)
"""

import os, time, json, argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from scipy.stats import ortho_group
from scipy.spatial import procrustes as scipy_procrustes
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
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, s): return s[..., 2:5] * self.sc

class RandomFixedEnc(nn.Module):
    def __init__(self, rotation_seed=0):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        R = ortho_group.rvs(3, random_state=rotation_seed).astype(np.float32)
        self.register_buffer("R", torch.from_numpy(R))
        print(f"  Random rotation (seed={rotation_seed}): det={np.linalg.det(R):.4f}, ortho_err={np.max(np.abs(R @ R.T - np.eye(3))):.2e}")
    def forward(self, s):
        return (s[..., 2:5] * self.sc) @ self.R.T

class FreeEnc(nn.Module):
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
                "p": p.detach(), "t": tgt.detach(),
                "emb": emb.detach()}


# --- Training ---
@dataclass
class Cfg:
    n_ep:int=200; max_steps:int=300; fs:int=5
    dim:int=3; hid:int=128; H:int=3
    epochs:int=50; bs:int=128; lr:float=3e-4; wd:float=1e-3
    lam:float=0.09; split:float=0.9; out:str="random_fixed_full_results"

SAVE_EPOCHS = [1, 5, 10, 25, 50]

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
def val_ep(m, dl, dev, collect_emb=False):
    m.eval(); tp,ts,n = 0,0,0; ps,ts_,embs = [],[],[]
    for s,a in dl:
        s,a = s.to(dev),a.to(dev); o = m(s,a)
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
        ps.append(o["p"].cpu()); ts_.append(o["t"].cpu())
        if collect_emb:
            embs.append(o["emb"].cpu())
    p,t = torch.cat(ps),torch.cat(ts_)
    per_axis = (p-t).pow(2).mean(0).tolist()
    emb_all = torch.cat(embs).numpy() if collect_emb else None
    return tp/n, ts/n, per_axis, emb_all

@torch.no_grad()
def collect_all_embeddings(m, ds, dev, batch_size=256):
    """Collect embeddings for ALL data (not just val)."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_emb, all_states = [], []
    m.eval()
    for s, a in dl:
        s = s.to(dev)
        emb = m.enc(s)  # [B, seq_len, dim]
        # Take the first state embedding (t=0) for alignment analysis
        all_emb.append(emb[:, 0].cpu().numpy())
        all_states.append(s[:, 0].cpu().numpy())
    return np.concatenate(all_emb), np.concatenate(all_states)

def make_encoder(mode, dim, rotation_seed=0):
    if mode == "prescribed":
        return PrescEnc()
    elif mode == "random_fixed":
        return RandomFixedEnc(rotation_seed=rotation_seed)
    elif mode == "free":
        return FreeEnc(5, dim, 64)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run(mode, eps, cfg, train_seed, dev, out_dir, rotation_seed=0):
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

    prefix = f"{mode}_seed{train_seed}"
    best,bep,hist = float("inf"),0,[]
    saved_embeddings = {}

    for ep in range(1,cfg.epochs+1):
        t0 = time.time()
        tp,ts = train_ep(mdl,tl,opt,cfg.lam,dev)

        # Check if we should collect embeddings this epoch
        collect = (ep in SAVE_EPOCHS) or (ep == cfg.epochs)
        vp,vs,axes,emb_val = val_ep(mdl,vl,dev,collect_emb=collect)
        sch.step()
        if vp<best: best,bep = vp,ep
        hist.append({"ep":ep,"tp":tp,"ts":ts,"vp":vp,"vs":vs,"ax":axes})

        if collect:
            # Collect ALL embeddings (full dataset) for alignment analysis
            emb_all, states_all = collect_all_embeddings(mdl, ds, dev)
            # Covariance matrix
            cov = np.cov(emb_all.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
            rank = np.sum(eigenvalues > eigenvalues[0] * 1e-5)

            saved_embeddings[ep] = {
                "emb": emb_all,
                "states": states_all,
                "cov": cov,
                "eigenvalues": eigenvalues,
                "rank": rank
            }

            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{', '.join(f'{v:.6f}' for v in axes)}] | {time.time()-t0:.1f}s | SAVED emb ({emb_all.shape[0]}x{emb_all.shape[1]}), rank={rank}, eig=[{', '.join(f'{v:.4e}' for v in eigenvalues)}]")
        else:
            if ep%10==0:
                print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{', '.join(f'{v:.6f}' for v in axes)}] | {time.time()-t0:.1f}s")

    print(f"  Best: {best:.6f} (ep {bep})")

    # Save model
    torch.save(mdl.state_dict(), out_dir / f"{prefix}_model.pt")

    # Save embeddings
    for ep, data in saved_embeddings.items():
        np.savez(out_dir / f"{prefix}_emb_ep{ep}.npz",
                 emb=data["emb"], states=data["states"],
                 cov=data["cov"], eigenvalues=data["eigenvalues"])

    # Save history
    with open(out_dir / f"{prefix}_history.json", "w") as fh:
        json.dump(hist, fh)

    return {
        "mode": mode, "seed": train_seed, "params": np_,
        "best": best, "bep": bep, "hist": hist,
        "embeddings": saved_embeddings,
        "rotation_seed": rotation_seed if mode == "random_fixed" else None
    }


def procrustes_analysis(emb_source, emb_target, name_source, name_target):
    """Full Procrustes analysis between two embedding sets."""
    print(f"\n  Procrustes: {name_source} → {name_target}")

    # Center both
    src = emb_source - emb_source.mean(0)
    tgt = emb_target - emb_target.mean(0)

    # Normalize
    src_norm = src / np.linalg.norm(src)
    tgt_norm = tgt / np.linalg.norm(tgt)

    # Optimal rotation
    R, scale = orthogonal_procrustes(src, tgt)
    aligned = src @ R

    # Residual after alignment
    residual = np.mean((aligned - tgt)**2)
    total_var = np.mean(tgt**2)
    r_squared = 1.0 - residual / total_var if total_var > 0 else 0.0

    # Per-axis R²
    per_axis_r2 = []
    for i in range(tgt.shape[1]):
        res_i = np.mean((aligned[:, i] - tgt[:, i])**2)
        var_i = np.mean(tgt[:, i]**2)
        r2_i = 1.0 - res_i / var_i if var_i > 0 else 0.0
        per_axis_r2.append(r2_i)

    # Disparity (scipy version for cross-check)
    _, _, disparity = scipy_procrustes(tgt_norm, src_norm)

    print(f"    R² = {r_squared:.6f}")
    print(f"    Per-axis R² = [{', '.join(f'{v:.4f}' for v in per_axis_r2)}]")
    print(f"    Disparity = {disparity:.6f}")
    print(f"    Rotation det = {np.linalg.det(R):.4f}")

    return {
        "r_squared": float(r_squared),
        "per_axis_r2": [float(v) for v in per_axis_r2],
        "disparity": float(disparity),
        "rotation_det": float(np.linalg.det(R)),
        "residual_mse": float(residual),
        "rotation_matrix": R.tolist()
    }


def eigenvalue_stability_analysis(results, mode, seed):
    """Analyze eigenvalue stability across epochs."""
    emb_data = results["embeddings"]
    epochs = sorted(emb_data.keys())
    print(f"\n  Eigenvalue stability: {mode} (seed={seed})")
    print(f"  {'Epoch':>6} | {'λ1':>10} | {'λ2':>10} | {'λ3':>10} | {'Rank':>4} | {'Condition':>10}")
    print(f"  {'-'*60}")

    stability = []
    for ep in epochs:
        eig = emb_data[ep]["eigenvalues"]
        rank = emb_data[ep]["rank"]
        cond = eig[0] / eig[-1] if eig[-1] > 0 else float("inf")
        print(f"  {ep:6d} | {eig[0]:10.4e} | {eig[1]:10.4e} | {eig[2]:10.4e} | {rank:4d} | {cond:10.2f}")
        stability.append({
            "epoch": ep,
            "eigenvalues": eig.tolist(),
            "rank": int(rank),
            "condition_number": float(cond)
        })

    # Compute drift between consecutive epochs
    if len(epochs) >= 2:
        print(f"\n  Eigenvalue drift (max ratio between consecutive epochs):")
        for i in range(1, len(epochs)):
            ep_prev, ep_curr = epochs[i-1], epochs[i]
            eig_prev = emb_data[ep_prev]["eigenvalues"]
            eig_curr = emb_data[ep_curr]["eigenvalues"]
            ratios = []
            for j in range(len(eig_prev)):
                if eig_prev[j] > 0:
                    ratios.append(eig_curr[j] / eig_prev[j])
            max_ratio = max(ratios) if ratios else 0
            min_ratio = min(ratios) if ratios else 0
            print(f"    ep {ep_prev}→{ep_curr}: max_ratio={max_ratio:.2f}×, min_ratio={min_ratio:.2f}×")

    return stability


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--epochs",type=int,default=50)
    pa.add_argument("--batch-size",type=int,default=128)
    pa.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    pa.add_argument("--embed-dim",type=int,default=3)
    pa.add_argument("--output-dir",type=str,default="random_fixed_full_results")
    pa.add_argument("--synthetic",action="store_true")
    pa.add_argument("--rotation-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = pa.parse_args()

    cfg = Cfg(n_ep=args.episodes, epochs=args.epochs, bs=args.batch_size,
              dim=args.embed_dim, out=args.output_dir)

    # Adjust SAVE_EPOCHS based on actual epochs
    global SAVE_EPOCHS
    SAVE_EPOCHS = [e for e in SAVE_EPOCHS if e <= cfg.epochs]
    if cfg.epochs not in SAVE_EPOCHS:
        SAVE_EPOCHS.append(cfg.epochs)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Training seeds: {args.seeds}")
    print(f"Rotation seeds: {args.rotation_seeds}")
    print(f"Save embeddings at epochs: {SAVE_EPOCHS}")

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
            r = run(mode, eps, cfg, seed, dev, out_dir, rotation_seed=rot_seed)
            all_results[mode].append(r)

    # ==========================================
    #  SUMMARY: PREDICTION ACCURACY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: PREDICTION ACCURACY")
    print(f"{'='*60}")

    for mode in modes:
        vals = [r["best"] for r in all_results[mode]]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {mode:15s}: {mean:.6f} ± {std:.6f}")

    p_mean = np.mean([r["best"] for r in all_results["prescribed"]])
    rf_mean = np.mean([r["best"] for r in all_results["random_fixed"]])
    f_mean = np.mean([r["best"] for r in all_results["free"]])

    print(f"\n  Ratios:")
    print(f"    free / prescribed:         {f_mean/p_mean:.1f}×")
    print(f"    free / random_fixed:       {f_mean/rf_mean:.1f}×")
    print(f"    random_fixed / prescribed: {rf_mean/p_mean:.2f}×")

    # ==========================================
    #  EIGENVALUE STABILITY ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  EIGENVALUE STABILITY ANALYSIS")
    print(f"{'='*60}")

    all_stability = {}
    for mode in modes:
        for r in all_results[mode]:
            key = f"{mode}_seed{r['seed']}"
            all_stability[key] = eigenvalue_stability_analysis(r, mode, r["seed"])

    # ==========================================
    #  PROCRUSTES ALIGNMENT ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  PROCRUSTES ALIGNMENT ANALYSIS")
    print(f"{'='*60}")

    final_epoch = cfg.epochs
    all_procrustes = {}

    for i, seed in enumerate(args.seeds):
        print(f"\n  --- Seed {seed} ---")
        presc_emb = all_results["prescribed"][i]["embeddings"][final_epoch]["emb"]
        rf_emb = all_results["random_fixed"][i]["embeddings"][final_epoch]["emb"]
        free_emb = all_results["free"][i]["embeddings"][final_epoch]["emb"]

        # Free → Prescribed
        key1 = f"free_to_prescribed_seed{seed}"
        all_procrustes[key1] = procrustes_analysis(free_emb, presc_emb, "free", "prescribed")

        # Free → Random Fixed
        key2 = f"free_to_random_fixed_seed{seed}"
        all_procrustes[key2] = procrustes_analysis(free_emb, rf_emb, "free", "random_fixed")

        # Random Fixed → Prescribed
        key3 = f"random_fixed_to_prescribed_seed{seed}"
        all_procrustes[key3] = procrustes_analysis(rf_emb, presc_emb, "random_fixed", "prescribed")

        # Drift analysis: free embeddings at epoch 1 vs epoch 50
        if 1 in all_results["free"][i]["embeddings"] and final_epoch in all_results["free"][i]["embeddings"]:
            free_ep1 = all_results["free"][i]["embeddings"][1]["emb"]
            free_ep50 = all_results["free"][i]["embeddings"][final_epoch]["emb"]
            key4 = f"free_drift_ep1_to_ep{final_epoch}_seed{seed}"
            all_procrustes[key4] = procrustes_analysis(free_ep1, free_ep50, "free@ep1", f"free@ep{final_epoch}")

        # Same for prescribed
        if 1 in all_results["prescribed"][i]["embeddings"] and final_epoch in all_results["prescribed"][i]["embeddings"]:
            presc_ep1 = all_results["prescribed"][i]["embeddings"][1]["emb"]
            presc_ep50 = all_results["prescribed"][i]["embeddings"][final_epoch]["emb"]
            key5 = f"prescribed_drift_ep1_to_ep{final_epoch}_seed{seed}"
            all_procrustes[key5] = procrustes_analysis(presc_ep1, presc_ep50, "prescribed@ep1", f"prescribed@ep{final_epoch}")

    # ==========================================
    #  FINAL INTERPRETATION
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  FINAL INTERPRETATION")
    print(f"{'='*60}")

    # Average Procrustes R² across seeds
    free_to_presc_r2 = np.mean([all_procrustes[f"free_to_prescribed_seed{s}"]["r_squared"] for s in args.seeds])
    free_to_rf_r2 = np.mean([all_procrustes[f"free_to_random_fixed_seed{s}"]["r_squared"] for s in args.seeds])
    rf_to_presc_r2 = np.mean([all_procrustes[f"random_fixed_to_prescribed_seed{s}"]["r_squared"] for s in args.seeds])

    print(f"\n  Average Procrustes R²:")
    print(f"    free → prescribed:         {free_to_presc_r2:.4f}")
    print(f"    free → random_fixed:       {free_to_rf_r2:.4f}")
    print(f"    random_fixed → prescribed: {rf_to_presc_r2:.4f}")

    # Drift R²
    drift_keys_free = [k for k in all_procrustes if "free_drift" in k]
    drift_keys_presc = [k for k in all_procrustes if "prescribed_drift" in k]

    if drift_keys_free:
        free_drift_r2 = np.mean([all_procrustes[k]["r_squared"] for k in drift_keys_free])
        print(f"    free self-drift (ep1→ep{final_epoch}):      {free_drift_r2:.4f}")
    if drift_keys_presc:
        presc_drift_r2 = np.mean([all_procrustes[k]["r_squared"] for k in drift_keys_presc])
        print(f"    prescribed self-drift (ep1→ep{final_epoch}): {presc_drift_r2:.4f}")

    print(f"\n  Interpretation:")
    if free_to_presc_r2 > 0.9:
        print(f"    → After Procrustes alignment, free ≈ prescribed (R²={free_to_presc_r2:.4f})")
        print(f"    → Free learns the SAME representation, just in rotated coordinates")
        print(f"    → Difference is ONLY gauge (coordinate choice)")
    elif free_to_presc_r2 > 0.5:
        print(f"    → Partial alignment (R²={free_to_presc_r2:.4f})")
        print(f"    → Free captures SOME of the same structure, but drift corrupts it")
    else:
        print(f"    → Poor alignment (R²={free_to_presc_r2:.4f})")
        print(f"    → Free does NOT learn the same representation")
        print(f"    → Problem is NOT just gauge — it is fundamental drift")

    if drift_keys_free and drift_keys_presc:
        if free_drift_r2 < 0.5 and presc_drift_r2 > 0.9:
            print(f"    → Free drifts heavily (R²={free_drift_r2:.4f}), prescribed stable (R²={presc_drift_r2:.4f})")
            print(f"    → Confirms: drift, not collapse, is the problem")

    # Save everything
    summary = {
        "experiment": "random_fixed_axes_full_with_procrustes",
        "data_seed": data_seed,
        "training_seeds": args.seeds,
        "rotation_seeds": args.rotation_seeds,
        "epochs": cfg.epochs,
        "save_epochs": SAVE_EPOCHS,
        "synthetic": args.synthetic,
        "prediction_results": {},
        "procrustes": all_procrustes,
        "eigenvalue_stability": all_stability,
    }
    for mode in modes:
        vals = [r["best"] for r in all_results[mode]]
        summary["prediction_results"][mode] = {
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

    with open(out_dir / "full_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  All results saved to {out_dir}/")
    print(f"  Files: model weights (.pt), embeddings (.npz), histories (.json), full_results.json")


if __name__ == "__main__":
    main()
