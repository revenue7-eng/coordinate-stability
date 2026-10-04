#!/usr/bin/env python3
# marker: E49 v1
"""E49. Run from the repository root:
  OMP_NUM_THREADS=1 python3 -u E49_completeness/code/run_e49.py --prereg <sha> --seeds ...

At 200 episodes, 90 epochs, per seed:
1. free_scaled stage 1 (JEPA on its own latent), identical to E45/E48 stage 1
   (gate: s1_hist equals E45 bit for bit). The frozen latent is probed for the
   state at the same step: targets (x_a, y_a, x_b, y_b)/512, sin theta, cos theta.
   Linear probe (least squares with bias) and MLP probe (5-64-64-6), fitted on
   the training split, R2 per target on the validation split.
   The same probes on prescribed_std are a sanity check (linear R2 near 1).
2. free_e2e: the same encoder class trained end to end on the E48 nonlinear
   common target from the start (encoder not frozen, no JEPA stage), same loop.
E48 cells supply prescribed_std and free_scaled (JEPA, frozen) on the same target.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

sys.path.insert(0, "E44_common_target/code")
import e44_lib as L  # noqa: E402

torch.set_num_threads(1)
SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
EPOCHS = 90
E45_CELLS = Path("E45_long_training/results/cells")
CELLS = Path("E49_completeness/results/cells")
CODE = ["E49_completeness/code/run_e49.py", "E49_completeness/code/analyze_e49.py"]
L.EPOCHS = EPOCHS
L.EPISODES = 200
NAMES = ["x_a", "y_a", "x_b", "y_b", "sin", "cos"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def require_prereg(commit):
    full = git("rev-parse", commit)
    subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], check=True)
    assert "### E49." in git("show", f"{full}:EXPERIMENTS.md"), "no E49 block at prereg commit"
    assert git("diff", "--name-only", full, "HEAD", "--", *CODE) == "", "code changed since prereg"
    assert git("status", "--porcelain", "--untracked-files=no", "--", *CODE) == "", "code not committed"
    return full


def g_target(s):  # identical to E48
    xa, ya, xb, yb = (s[..., i] / 512 for i in range(4))
    th = s[..., 4]
    dx, dy = xa - xb, ya - yb
    c, sn = torch.cos(th), torch.sin(th)
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    return torch.stack([dist, sn, c, dx * c + dy * sn, -dx * sn + dy * c], -1)


_WM44 = L.WorldModel


class WM48(_WM44):
    def forward(self, st, a):
        if L._E44["target"] == "self":
            return super().forward(st, a)
        emb = self.enc(st)
        ctx = emb[:, :self.H]
        with torch.no_grad():
            tgt = g_target(st[:, self.H])
        p = self.pr(ctx, self.ae(a[:, :self.H]))
        return {"pl": Fn.mse_loss(p, tgt), "sl": self.sig(emb.transpose(0, 1))}


class PrescribedStd(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, x):
        return (L.make_prescribed_features(x, 5) - self.mu) / self.sd


def states(dl_dataset):
    out = []
    for s, _ in DataLoader(dl_dataset, batch_size=4096, shuffle=False):
        out.append(s.reshape(-1, 5))
    return torch.cat(out)


def probe_targets(s):
    th = s[:, 4]
    return torch.cat([s[:, :4] / 512, torch.sin(th)[:, None], torch.cos(th)[:, None]], 1)


def r2(pred, y):
    return (1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()


@torch.no_grad()
def embed(enc, s):
    enc.eval()
    return enc(s)


def linear_probe(ztr, ytr, zva, yva):
    A = torch.cat([ztr, torch.ones(len(ztr), 1)], 1)
    W = torch.linalg.lstsq(A, ytr).solution
    return r2(torch.cat([zva, torch.ones(len(zva), 1)], 1) @ W, yva)


def mlp_probe(ztr, ytr, zva, yva, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(ztr.shape[1], 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, ytr.shape[1]))
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = len(ztr)
    for _ in range(100):
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            loss = Fn.mse_loss(m((ztr[b] - mu) / sd), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return r2(m((zva - mu) / sd), yva)


def probes(enc, tl, vl, seed):
    s_tr, s_va = states(tl.dataset), states(vl.dataset)
    ztr, zva = embed(enc, s_tr), embed(enc, s_va)
    ytr, yva = probe_targets(s_tr), probe_targets(s_va)
    return {"linear_r2": linear_probe(ztr, ytr, zva, yva), "mlp_r2": mlp_probe(ztr, ytr, zva, yva, seed),
            "targets": NAMES}


_val = L.val_loss
_prog = {"tag": "", "ep": 0, "t0": 0.0}


def _progress(mdl, vl):
    v = _val(mdl, vl)
    _prog["ep"] += 1
    print(f"{_prog['tag']} epoch {(_prog['ep'] - 1) % EPOCHS + 1:2d}/{EPOCHS} {time.time() - _prog['t0']:6.0f} s", flush=True)
    return v


L.val_loss = _progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for s in a.seeds:
        assert s in SEEDS, s
    full = require_prereg(a.prereg)
    CELLS.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        f = CELLS / f"seed_{s}.json"
        if f.exists():
            print(f"seed {s}: exists, skipped", flush=True)
            continue
        rec = {"seed": s, "prereg": full, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "epochs": EPOCHS}
        # 1. JEPA stage 1, as run_arm stage 1
        tl, vl = L.make_data(s)
        _prog.update(tag=f"seed {s} free_scaled jepa", ep=0, t0=time.time())
        torch.manual_seed(s)
        enc = L.FreeEncoderScaled(5)
        L.set_target("self")
        h1 = L._e44_train(L.WorldModel(enc, 5), tl, vl, EPOCHS)
        e45 = json.loads((E45_CELLS / f"seed_{s}.json").read_text())
        rec["jepa"] = {"s1_hist": h1, "s1_equals_e45": h1 == e45["arms"]["free_scaled"]["s1_hist"]}
        rec["jepa"]["probes"] = probes(enc, tl, vl, s)
        # sanity: prescribed_std
        s_tr = states(tl.dataset)
        f5 = L.make_prescribed_features(s_tr, 5)
        rec["prescribed_std_probes"] = probes(PrescribedStd(f5.mean(0), f5.std(0)), tl, vl, s)
        # 2. end to end on the nonlinear target
        tl, vl = L.make_data(s)
        _prog.update(tag=f"seed {s} free_e2e", ep=0, t0=time.time())
        torch.manual_seed(s)
        enc2 = L.FreeEncoderScaled(5)
        L.set_target("external")
        try:
            h2 = L._e44_train(WM48(enc2, 5), tl, vl, EPOCHS)
        finally:
            L.set_target("self")
        rec["e2e"] = {"hist": h2, "final": h2[-1], "best": min(h2), "probes": probes(enc2, tl, vl, s)}
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
