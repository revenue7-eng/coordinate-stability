#!/usr/bin/env python3
"""
B1 -- does the free encoder of E34 implicitly represent the room geometry?

Question (registry question 8): the free ImpalaEncoder sees only pixels. Does its
latent linearly encode wall_x / door_y, or does it solve the task some other way?

Metric: held-out R^2 of a ridge regression z_free -> wall_x and z_free -> door_y.
Thresholds, fixed in advance (PLAN_2026-08-20, priority 1):
    R^2 > 0.7  -- free solves it through an explicit representation of the wall
    R^2 < 0.3  -- through something else (predictor memory, indirect cues)
    in between -- undetermined, needs separate analysis

Design notes, each one verified against an artifact before this script was written:

  * WallDataset.__getitem__ ignores its index and draws a fresh episode on every
    call (dot_dataset.py:63). There is therefore no train/test leakage by
    construction, and no way to reproduce the exact episodes E34 trained on --
    init_data is called before setup_seed in run_experiment_v3_windows.py:370-373,
    so that draw was never seeded. The correct claim about this measurement is
    "on a fresh sample from the E34 configuration", not "on E34's training data".

  * wall_x / door_y come out of the sample as scalars per episode (WallSample,
    wall_dataset.py:18-23) and bypass the normalizer -- they stay in raw pixel
    units. Only states and locations are normalized.

  * The latent is 512-dimensional (encoder.final_ln, checkpoint), not 20. With
    n/d around 8 at the default sample size, ridge with cross-validated alpha is
    load-bearing, not decoration: plain OLS would fit the train set perfectly and
    generalize to noise.

  * Architecture hyperparameters are inferred from the checkpoint tensor shapes
    rather than retyped from the config, and the state dict is then loaded with
    strict=True. If the inference is wrong the script dies instead of silently
    measuring a differently-shaped network.

Seeding: SEED fixes our own Monte Carlo draw (which episodes, which permutations)
so the numbers are reproducible. It does not freeze anything inside the model.

Run:
    PYTHONPATH=<path to eb_jepa_free> \
    ~/.venv-b1/bin/python b1_latent_physics.py \
        --ckpt <path>/eb_jepa_free/results/free/latest.pth.tar \
        --data-config <path>/eb_jepa_free/eb_jepa/datasets/two_rooms/data_config.yaml
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

from eb_jepa.architectures import ImpalaEncoder
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

SEED = 42
ALPHAS = np.logspace(-3, 5, 25)
N_PERM = 20
FRAMES = (0, 8)  # geometry is constant within an episode; t is a stability control


# ----------------------------------------------------------------- model

def build_encoder_from_checkpoint(ckpt_path):
    """Infer ImpalaEncoder geometry from tensor shapes, build, load strict."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_sd = {k[len("encoder."):]: v
              for k, v in ck["model_state_dict"].items()
              if k.startswith("encoder.")}
    if not enc_sd:
        sys.exit("FAIL: no 'encoder.' tensors in model_state_dict")

    # stack_blocks.<i>.initial_conv.weight has shape (out_ch, in_ch, 3, 3)
    stack_sizes = []
    i = 0
    while f"stack_blocks.{i}.initial_conv.weight" in enc_sd:
        stack_sizes.append(enc_sd[f"stack_blocks.{i}.initial_conv.weight"].shape[0])
        i += 1
    if not stack_sizes:
        sys.exit("FAIL: could not infer stack_sizes")

    dobs = enc_sd["stack_blocks.0.initial_conv.weight"].shape[1]
    mlp_out, flattened = enc_sd["mlp.weight"].shape
    has_final_ln = "final_ln.weight" in enc_sd
    img = 65  # from data_config; asserted against the real sample below

    enc = ImpalaEncoder(
        width=1,
        stack_sizes=tuple(stack_sizes),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=dobs,
        final_ln=has_final_ln,
        mlp_output_dim=mlp_out,
        input_shape=(dobs, img, img),
    )
    missing, unexpected = [], []
    try:
        enc.load_state_dict(enc_sd, strict=True)
    except RuntimeError as e:
        sys.exit(f"FAIL: strict load_state_dict rejected the inferred architecture\n{e}")
    enc.eval()

    n_params = sum(v.numel() for v in enc_sd.values())
    print(f"[ckpt] epoch={ck.get('epoch')} step={ck.get('step')}")
    print(f"[ckpt] stack_sizes={tuple(stack_sizes)} dobs={dobs} "
          f"flattened={flattened} latent={mlp_out} final_ln={has_final_ln}")
    print(f"[ckpt] encoder params={n_params}  (E34 recorded 1426096)")
    print(f"[ckpt] strict load OK  ({len(missing)} missing, {len(unexpected)} unexpected)")
    return enc, dobs, mlp_out, ck.get("epoch")


# ----------------------------------------------------------------- data

def build_dataset(yaml_path):
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(WallDatasetConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    dropped = sorted(set(raw) - valid)
    cfg = WallDatasetConfig(**kwargs)
    print(f"[data] config fields used={len(kwargs)} ignored={dropped}")
    print(f"[data] fix_wall={cfg.fix_wall} img_size={cfg.img_size} "
          f"normalize={cfg.normalize} sample_length={cfg.sample_length}")
    if cfg.fix_wall:
        sys.exit("FAIL: fix_wall is True -- geometry is constant, R^2 is undefined")
    return WallDataset(cfg)


def collect(ds, encoder, n, dobs, latent_dim, frames, tag):
    """Draw n fresh episodes; return {t: Z}, wall_x, door_y."""
    Z = {t: np.empty((n, latent_dim), dtype=np.float64) for t in frames}
    wx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        s = ds[i]
        states = s.states
        if states.shape[0] != dobs:
            sys.exit(f"FAIL: expected (C,T,H,W) with C={dobs}, got {tuple(states.shape)}")
        with torch.no_grad():
            for t in frames:
                # encoder wants [B, C, T, H, W]; keep the T axis, length 1
                x = states[:, t:t + 1].unsqueeze(0).float()
                Z[t][i] = encoder(x).reshape(-1).numpy()
        wx[i] = float(s.wall_x.reshape(-1)[0])
        dy[i] = float(s.door_y.reshape(-1)[0])
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{tag}] {i+1}/{n}  {el:.0f}s  ({el/(i+1)*1000:.0f} ms/ep)")
    return Z, wx, dy


# ----------------------------------------------------------------- fit

def fit_r2(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return r2_score(yte, m.predict(sc.transform(Xte))), m.alpha_


def permutation_floor(Xtr, ytr, Xte, yte, rng):
    """Pipeline floor: refit on shuffled train labels, score against real test."""
    out = []
    for _ in range(N_PERM):
        out.append(fit_r2(Xtr, rng.permutation(ytr), Xte, yte)[0])
    return float(np.mean(out)), float(np.std(out)), float(np.max(out))


def verdict(r2):
    if r2 > 0.7:
        return "ABOVE 0.7 -- explicit representation of the geometry"
    if r2 < 0.3:
        return "BELOW 0.3 -- geometry is not linearly present in the latent"
    return "BETWEEN 0.3 and 0.7 -- UNDETERMINED, needs separate analysis"


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random as _r
    _r.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("B1 -- R^2(z_free -> wall_x, door_y),  thresholds 0.7 / 0.3 fixed in advance")
    print("=" * 72)

    encoder, dobs, latent_dim, epoch = build_encoder_from_checkpoint(args.ckpt)
    ds = build_dataset(args.data_config)

    print(f"[draw] {args.n_train} train + {args.n_test} test fresh episodes, "
          f"frames t={FRAMES}, seed={SEED}")
    Ztr, wxtr, dytr = collect(ds, encoder, args.n_train, dobs, latent_dim, FRAMES, "train")
    Zte, wxte, dyte = collect(ds, encoder, args.n_test, dobs, latent_dim, FRAMES, "test")

    # Direct check that the geometry actually varies -- by artifact, not by config.
    print("\n[labels] measured spread across episodes (raw pixel units)")
    for nm, a in (("wall_x", wxtr), ("door_y", dytr)):
        print(f"  {nm}: n_unique={len(np.unique(a))} min={a.min():.2f} "
              f"max={a.max():.2f} sd={a.std():.3f}")
        if a.std() < 1e-6:
            sys.exit(f"FAIL: {nm} is constant -- R^2 undefined")

    print(f"\n[fit] RidgeCV, alphas {ALPHAS[0]:.0e}..{ALPHAS[-1]:.0e}, "
          f"n/d = {args.n_train}/{latent_dim} = {args.n_train/latent_dim:.1f}")

    results = {}
    for t in FRAMES:
        print(f"\n--- frame t={t} " + "-" * 55)
        for nm, ytr, yte in (("wall_x", wxtr, wxte), ("door_y", dytr, dyte)):
            r2, alpha = fit_r2(Ztr[t], ytr, Zte[t], yte)
            fm, fs, fmax = permutation_floor(Ztr[t], ytr, Zte[t], yte, rng)
            results[(t, nm)] = r2
            print(f"  {nm:8s} R2={r2:+.4f}  alpha={alpha:.3g}")
            print(f"           permutation floor {fm:+.4f} +/- {fs:.4f} "
                  f"(max {fmax:+.4f}, {N_PERM} draws)")
            print(f"           above floor: {r2 - fm:+.4f}")
            if t == FRAMES[0]:
                print(f"           VERDICT: {verdict(r2)}")

    print("\n" + "=" * 72)
    print("SUMMARY  (primary frame t=%d, checkpoint epoch %s)" % (FRAMES[0], epoch))
    for nm in ("wall_x", "door_y"):
        prim = results[(FRAMES[0], nm)]
        others = ", ".join(f"t={t}: {results[(t, nm)]:+.4f}" for t in FRAMES[1:])
        print(f"  {nm:8s} R2={prim:+.4f}   [{others}]   {verdict(prim)}")
    print("=" * 72)
    print("NOT measured here: prescribed_2 as a structural zero (its latent cannot")
    print("contain the wall by construction). That checkpoint lives in a separate")
    print("archive and is a second pass.")


if __name__ == "__main__":
    main()
