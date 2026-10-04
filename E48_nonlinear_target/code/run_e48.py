#!/usr/bin/env python3
# marker: E48 v1
"""E48. Run from the repository root:
  OMP_NUM_THREADS=1 python3 -u E48_nonlinear_target/code/run_e48.py --prereg <sha> --seeds ...

As E47 (prescribed_std stage 2 only, free_scaled both stages, 90 epochs), at 25
and 200 episodes, with a common target that is nonlinear in the prescribed
coordinates: g(s_{t+3}) = (dist(agent, block), sin theta, cos theta, u, v),
where (u, v) is the agent position relative to the block in the block's frame.
Positions are divided by 512. Stage 1 is unchanged, so free_scaled stage 1 at
200 episodes must equal E45 free_scaled s1_hist bit for bit (gate).
The E44 library is imported, not modified.
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
SIZES = (25, 200)
E45_CELLS = Path("E45_long_training/results/cells")
CELLS = Path("E48_nonlinear_target/results/cells")
CODE = ["E48_nonlinear_target/code/run_e48.py", "E48_nonlinear_target/code/analyze_e48.py"]
L.EPOCHS = EPOCHS


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def require_prereg(commit):
    full = git("rev-parse", commit)
    subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], check=True)
    assert "### E48." in git("show", f"{full}:EXPERIMENTS.md"), "no E48 block at prereg commit"
    assert git("diff", "--name-only", full, "HEAD", "--", *CODE) == "", "code changed since prereg"
    assert git("status", "--porcelain", "--untracked-files=no", "--", *CODE) == "", "code not committed"
    return full


def g_target(s):
    xa, ya, xb, yb = (s[..., i] / 512 for i in range(4))
    th = s[..., 4]
    dx, dy = xa - xb, ya - yb
    c, sn = torch.cos(th), torch.sin(th)
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    u = dx * c + dy * sn
    v = -dx * sn + dy * c
    return torch.stack([dist, sn, c, u, v], -1)


_WM44 = L.WorldModel


class WM48(_WM44):
    def forward(self, st, a):
        if L._E44["target"] == "self":
            return super().forward(st, a)
        emb = self.enc(st)
        ctx = emb[:, :self.H]
        with torch.no_grad():
            tgt = g_target(st[:, self.H])
        aem = self.ae(a[:, :self.H])
        p = self.pr(ctx, aem)
        return {"pl": Fn.mse_loss(p, tgt), "sl": self.sig(emb.transpose(0, 1))}


L.WorldModel = WM48  # run_arm resolves WorldModel from the module at call time


class PrescribedStd(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def forward(self, x):
        return (L.make_prescribed_features(x, 5) - self.mu) / self.sd


def train_stats(tl):
    fs = []
    for s, _ in DataLoader(tl.dataset, batch_size=4096, shuffle=False):
        fs.append(L.make_prescribed_features(s, 5).reshape(-1, 5))
    f = torch.cat(fs)
    return f.mean(0), f.std(0)


_train = L._e44_train
_checks = []


def _checked_train(mdl, tl, vl, epochs):
    if L._E44["target"] != "external":
        return _train(mdl, tl, vl, epochs)
    assert isinstance(mdl, WM48)
    before = {k: v.detach().clone() for k, v in mdl.enc.state_dict().items()}
    h = _train(mdl, tl, vl, epochs)
    after = mdl.enc.state_dict()
    _checks.append(set(before) == set(after) and all(torch.equal(before[k], after[k]) for k in before))
    return h


L._e44_train = _checked_train
_val = L.val_loss
_prog = {"tag": "", "ep": 0, "t0": 0.0}


def _progress(mdl, vl):
    v = _val(mdl, vl)
    _prog["ep"] += 1
    st = 1 if L._E44["target"] == "self" else 2
    print(f"{_prog['tag']} stage {st} epoch {(_prog['ep'] - 1) % EPOCHS + 1:2d}/{EPOCHS} "
          f"{time.time() - _prog['t0']:6.0f} s", flush=True)
    return v


L.val_loss = _progress


def stage2_only(enc, tl, vl, seed):
    torch.manual_seed(seed)
    L.set_target("external")
    try:
        mdl = L.WorldModel(enc, 5)
        _checks.clear()
        hist = L._e44_train(mdl, tl, vl, EPOCHS)
    finally:
        L.set_target("self")
    assert len(_checks) == 1
    return {"s2_hist": hist, "s2_final": hist[-1], "s2_best": min(hist),
            "encoder_unchanged_in_stage2": _checks[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for s in a.seeds:
        assert s in SEEDS, s
    full = require_prereg(a.prereg)
    # target sanity: shape, finiteness, unit circle
    x = torch.tensor([[100., 200., 300., 250., 1.0]])
    t = g_target(x)
    assert t.shape == (1, 5) and torch.isfinite(t).all()
    assert abs(float(t[0, 1] ** 2 + t[0, 2] ** 2) - 1) < 1e-6
    CELLS.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        f = CELLS / f"seed_{s}.json"
        if f.exists():
            print(f"seed {s}: exists, skipped", flush=True)
            continue
        rec = {"seed": s, "prereg": full, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "epochs": EPOCHS, "sizes": {}}
        for n in SIZES:
            L.EPISODES = n
            rec["sizes"][str(n)] = {}
            tl, vl = L.make_data(s)
            assert len(tl) >= 5, (n, len(tl))
            _prog.update(tag=f"seed {s} n{n} free_scaled", ep=0, t0=time.time())
            _checks.clear()
            r = L.run_arm("free_scaled", tl, vl, s, stage2=True, epochs=EPOCHS)
            r["encoder_unchanged_in_stage2"] = len(_checks) == 1 and _checks[0]
            if n == 200:
                e45 = json.loads((E45_CELLS / f"seed_{s}.json").read_text())
                r["s1_equals_e45"] = r["s1_hist"] == e45["arms"]["free_scaled"]["s1_hist"]
            rec["sizes"][str(n)]["free_scaled"] = r
            tl, vl = L.make_data(s)
            mu, sd = train_stats(tl)
            _prog.update(tag=f"seed {s} n{n} prescribed_std", ep=0, t0=time.time())
            r = stage2_only(PrescribedStd(mu, sd), tl, vl, s)
            rec["sizes"][str(n)]["prescribed_std"] = r
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
