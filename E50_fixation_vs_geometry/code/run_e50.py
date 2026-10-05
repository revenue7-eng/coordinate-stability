#!/usr/bin/env python3
# marker: E50 v1
"""E50. Run from the repository root:
  OMP_NUM_THREADS=1 python3 -u E50_fixation_vs_geometry/code/run_e50.py --prereg <sha> --seeds ...

200 episodes, E48 nonlinear common target, per seed:
1. fixed_warped: frozen, complete, nonlinearly warped encoder of the state.
   f = standardised prescribed features; h = f Q1; h += A tanh(B h); h = h Q2;
   h += A tanh(B h); h = h Q3; output standardised on the training split.
   A = 1.5, B = 2 (each step strictly monotone, so the map is invertible).
   Q1..Q3 random orthogonal from a private generator seeded 50000 + seed.
   Stage 2 only, 90 epochs, as prescribed_std in E48. Probed like E49.
2. free_e2e_fair: FreeEncoderScaled trained end to end on the target for 180
   epochs (the total of JEPA stage 1 + stage 2), SIGReg weight 0.
prescribed_std and frozen JEPA free_scaled come from E48 cells.
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
CELLS = Path("E50_fixation_vs_geometry/results/cells")
CODE = ["E50_fixation_vs_geometry/code/run_e50.py", "E50_fixation_vs_geometry/code/analyze_e50.py"]
L.EPISODES = 200
A_W, B_W = 1.5, 2.0
_EP = {"n": 90}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def require_prereg(commit):
    full = git("rev-parse", commit)
    subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], check=True)
    assert "### E50." in git("show", f"{full}:EXPERIMENTS.md"), "no E50 block at prereg commit"
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


class WM48(L.WorldModel):
    def forward(self, st, a):
        if L._E44["target"] == "self":
            return super().forward(st, a)
        emb = self.enc(st)
        with torch.no_grad():
            tgt = g_target(st[:, self.H])
        p = self.pr(emb[:, :self.H], self.ae(a[:, :self.H]))
        return {"pl": Fn.mse_loss(p, tgt), "sl": self.sig(emb.transpose(0, 1))}


def orth(g):
    q, r = torch.linalg.qr(torch.randn(5, 5, generator=g))
    return q * torch.sign(torch.diagonal(r))


class FixedWarped(nn.Module):
    def __init__(self, mu, sd, seed):
        super().__init__()
        g = torch.Generator().manual_seed(50000 + seed)
        for k in ("Q1", "Q2", "Q3"):
            self.register_buffer(k, orth(g))
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.register_buffer("omu", torch.zeros(5)); self.register_buffer("osd", torch.ones(5))

    def raw(self, x):
        h = ((L.make_prescribed_features(x, 5) - self.mu) / self.sd) @ self.Q1
        h = h + A_W * torch.tanh(B_W * h)
        h = h @ self.Q2
        h = h + A_W * torch.tanh(B_W * h)
        return h @ self.Q3

    def forward(self, x):
        return (self.raw(x) - self.omu) / self.osd


def states(ds):
    return torch.cat([s.reshape(-1, 5) for s, _ in DataLoader(ds, batch_size=4096, shuffle=False)])


def probe_targets(s):
    th = s[:, 4]
    return torch.cat([s[:, :4] / 512, torch.sin(th)[:, None], torch.cos(th)[:, None]], 1)


def r2(p, y):
    return (1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()


def probes(enc, tl, vl, seed):
    s_tr, s_va = states(tl.dataset), states(vl.dataset)
    with torch.no_grad():
        ztr, zva = enc(s_tr), enc(s_va)
    ytr, yva = probe_targets(s_tr), probe_targets(s_va)
    A = torch.cat([ztr, torch.ones(len(ztr), 1)], 1)
    W = torch.linalg.lstsq(A, ytr).solution
    lin = r2(torch.cat([zva, torch.ones(len(zva), 1)], 1) @ W, yva)
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 6))
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(100):
        perm = torch.randperm(len(ztr))
        for i in range(0, len(ztr), 1024):
            b = perm[i:i + 1024]
            loss = Fn.mse_loss(m((ztr[b] - mu) / sd), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mlp = r2(m((zva - mu) / sd), yva)
    return {"linear_r2": lin, "mlp_r2": mlp}


_val = L.val_loss
_prog = {"tag": "", "ep": 0, "t0": 0.0}


def _progress(mdl, vl):
    v = _val(mdl, vl)
    _prog["ep"] += 1
    print(f"{_prog['tag']} epoch {_prog['ep']:3d}/{_EP['n']} {time.time() - _prog['t0']:6.0f} s", flush=True)
    return v


L.val_loss = _progress


def train(mdl, tl, vl, epochs, sig_w):
    """The E44 loop (AdamW 3e-4, wd 1e-3, clip 1.0) with a given SIGReg weight."""
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-3)
    hist = []
    for _ in range(epochs):
        mdl.train()
        for s, a in tl:
            o = mdl(s, a)
            l = o["pl"] + sig_w * o["sl"]
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
        hist.append(L.val_loss(mdl, vl))
    return hist


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
        rec = {"seed": s, "prereg": full, "torch": torch.__version__, "threads": torch.get_num_threads()}
        # 1. fixed_warped, stage 2 only
        tl, vl = L.make_data(s)
        s_tr = states(tl.dataset)
        f5 = L.make_prescribed_features(s_tr, 5)
        enc = FixedWarped(f5.mean(0), f5.std(0), s)
        with torch.no_grad():
            h = enc.raw(s_tr)
        enc.omu.copy_(h.mean(0)); enc.osd.copy_(h.std(0))
        before = {k: v.clone() for k, v in enc.state_dict().items()}
        _EP["n"] = 90; _prog.update(tag=f"seed {s} fixed_warped", ep=0, t0=time.time())
        torch.manual_seed(s)
        L.set_target("external")
        try:
            hw = train(WM48(enc, 5), tl, vl, 90, 0.09)
        finally:
            L.set_target("self")
        unchanged = all(torch.equal(before[k], v) for k, v in enc.state_dict().items())
        rec["fixed_warped"] = {"s2_hist": hw, "s2_final": hw[-1], "encoder_unchanged": unchanged,
                               "probes": probes(enc, tl, vl, s)}
        # 2. free_e2e_fair
        tl, vl = L.make_data(s)
        _EP["n"] = 180; _prog.update(tag=f"seed {s} free_e2e_fair", ep=0, t0=time.time())
        torch.manual_seed(s)
        enc2 = L.FreeEncoderScaled(5)
        L.set_target("external")
        try:
            he = train(WM48(enc2, 5), tl, vl, 180, 0.0)
        finally:
            L.set_target("self")
        rec["free_e2e_fair"] = {"hist": he, "final": he[-1], "best": min(he)}
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
