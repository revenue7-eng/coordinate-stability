#!/usr/bin/env python3
"""
B1 control -- structural zero on the prescribed_2 encoder.

Purpose. B1 measured R^2(z_free -> wall_x) = 0.9691 on held-out episodes. That
number is only trustworthy if the pipeline itself cannot manufacture it. The
permutation floor already showed the fit does not survive label shuffling; this
is the complementary check, on the input side.

The prescribed_2 encoder is an MLP over two numbers, the agent's (x, y). Its
latent cannot contain the wall in any richer form than those two numbers allow.
So it gives a ceiling that is defined by construction rather than by shuffling.

Important, and the reason this script reports two numbers instead of one:
in Two Rooms the agent's own position is correlated with the wall by design --
trajectories are generated relative to it (cross_wall_rate 0.35, start/goal via
generate_cross_wall_points(wall_locs)). A nonzero R^2 from prescribed is
therefore expected and is NOT evidence of leakage. What matters is the
comparison:

    R^2(z_prescribed -> wall_x)   what the latent yields
    R^2(locations    -> wall_x)   what its own input already contains

  close together  -> honest: the latent knows no more than what it was fed,
                     and the free number stands
  latent >> input -> the pipeline is inventing signal, and 0.9691 is suspect

Checkpoint note: this checkpoint carries epoch 0, step 1561, not the epoch 11
that produced planning SR = 0%. For this control that is not a defect -- the
argument is about what the input can carry, not about what training achieved.
If anything, untrained weights make the check stricter.

Run:
    PYTHONPATH=<eb_jepa_free> ~/.venv-b1/bin/python b1_control_prescribed.py \
        --ckpt <...>/pa_prescribed_seed1_seed1/latest.pth.tar \
        --data-config <eb_jepa_free>/eb_jepa/datasets/two_rooms/data_config.yaml
"""

import argparse
import dataclasses
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig
from experiments.prescribed_axes.prescribed_encoder import PrescribedEncoder

SEED = 42
ALPHAS = np.logspace(-3, 5, 25)
N_PERM = 20
FRAME = 0  # same primary frame as B1


def build_encoder(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[len("encoder."):]: v
          for k, v in ck["model_state_dict"].items()
          if k.startswith("encoder.")}
    if not sd:
        sys.exit("FAIL: no 'encoder.' tensors in model_state_dict")

    pres_dim = sd["projection.0.weight"].shape[1]
    hidden = sd["projection.0.weight"].shape[0]
    out_dim = sd["projection.4.weight"].shape[0]
    has_ln = "final_ln.weight" in sd

    enc = PrescribedEncoder(prescribed_dim=pres_dim, output_dim=out_dim,
                            hidden_dim=hidden, final_ln=has_ln)
    try:
        enc.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        sys.exit(f"FAIL: strict load rejected the inferred architecture\n{e}")
    enc.eval()
    print(f"[ckpt] epoch={ck.get('epoch')} step={ck.get('step')}")
    print(f"[ckpt] prescribed_dim={pres_dim} hidden={hidden} latent={out_dim} "
          f"final_ln={has_ln}")
    print(f"[ckpt] params={sum(v.numel() for v in sd.values())} "
          f"(E34 recorded 199168)")
    print("[ckpt] strict load OK")
    return enc, pres_dim, out_dim


def build_dataset(yaml_path):
    raw = yaml.safe_load(open(yaml_path))
    valid = {f.name for f in dataclasses.fields(WallDatasetConfig)}
    cfg = WallDatasetConfig(**{k: v for k, v in raw.items() if k in valid})
    if cfg.fix_wall:
        sys.exit("FAIL: fix_wall is True")
    print(f"[data] fix_wall={cfg.fix_wall} normalize={cfg.normalize} "
          f"sample_length={cfg.sample_length}")
    return WallDataset(cfg)


def collect(ds, enc, n, latent_dim, tag):
    Z = np.empty((n, latent_dim))
    L1 = np.empty((n, 2))    # locations at FRAME -- the encoder's own input
    Lall = None
    wx = np.empty(n)
    dy = np.empty(n)
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        loc = s.locations                       # (2, T)
        if Lall is None:
            Lall = np.empty((n, loc.shape[0] * loc.shape[1]))
        with torch.no_grad():
            # encoder wants observations [B,C,T,H,W] (ignored) and locations [B,2,T]
            obs = s.states[:, FRAME:FRAME + 1].unsqueeze(0).float()
            lc = loc[:, FRAME:FRAME + 1].unsqueeze(0).float()
            Z[i] = enc(obs, locations=lc).reshape(-1).numpy()
        L1[i] = loc[:, FRAME].numpy()
        Lall[i] = loc.reshape(-1).numpy()
        wx[i] = float(s.wall_x.reshape(-1)[0])
        dy[i] = float(s.door_y.reshape(-1)[0])
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{tag}] {i+1}/{n}  {el:.0f}s")
    return Z, L1, Lall, wx, dy


def fit_r2(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return r2_score(yte, m.predict(sc.transform(Xte))), m.alpha_


def floor(Xtr, ytr, Xte, yte, rng):
    v = [fit_r2(Xtr, rng.permutation(ytr), Xte, yte)[0] for _ in range(N_PERM)]
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    a = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    import random as _r; _r.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("B1 CONTROL -- prescribed_2 as a structural zero")
    print("=" * 72)
    enc, pres_dim, latent = build_encoder(a.ckpt)
    ds = build_dataset(a.data_config)
    print(f"[draw] {a.n_train} train + {a.n_test} test, frame t={FRAME}, seed={SEED}")

    Ztr, L1tr, Latr, wxtr, dytr = collect(ds, enc, a.n_train, latent, "train")
    Zte, L1te, Late, wxte, dyte = collect(ds, enc, a.n_test, latent, "test")

    for nm, y_tr, y_te in (("wall_x", wxtr, wxte), ("door_y", dytr, dyte)):
        print(f"\n--- {nm} " + "-" * 58)
        r_lat, al = fit_r2(Ztr, y_tr, Zte, y_te)
        f_lat, s_lat = floor(Ztr, y_tr, Zte, y_te, rng)
        r_in, _ = fit_r2(L1tr, y_tr, L1te, y_te)
        r_all, _ = fit_r2(Latr, y_tr, Late, y_te)
        print(f"  latent z_prescribed ({latent}d)   R2={r_lat:+.4f}  alpha={al:.3g}")
        print(f"     permutation floor            {f_lat:+.4f} +/- {s_lat:.4f}")
        print(f"  its own input, locations[t] (2d) R2={r_in:+.4f}")
        print(f"  full trajectory locations ({Latr.shape[1]}d) R2={r_all:+.4f}")
        gap = r_lat - r_in
        print(f"  latent minus own input: {gap:+.4f}")
        if gap > 0.10:
            print("  ** latent exceeds its own input by a wide margin --")
            print("  ** the pipeline may be manufacturing signal. Investigate.")
        else:
            print("  OK: the latent yields no more than what it was fed.")

    print("\n" + "=" * 72)
    print("Read against B1: free wall_x R2=0.9691, door_y R2=0.2109 (t=0, n=4000).")
    print("=" * 72)


if __name__ == "__main__":
    main()
