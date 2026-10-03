#!/usr/bin/env python3
# marker: E46 v1
"""E46. Run from the repository root:
  OMP_NUM_THREADS=1 python3 -u E46_low_data_basis/code/run_e46.py --prereg <sha> --seeds 42 123 ...

Part A (Г-a): prescribed and free_scaled, both stages as in E44, at 25 and 50
episodes (30 epochs per stage, as E44).
Part B (Г-b): at 200 episodes, stage 2 only (encoder fixed, fresh predictor on
the common target):
  prescribed_gate   plain prescribed; must equal E44 prescribed s2_hist bit for bit
  prescribed_sincos (x_a, y_a, x_b, y_b, sin, cos), range-normalised positions
  prescribed_std    the five prescribed features standardised with mean and std
                    of the training split of that seed
The E44 library is imported, not modified.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "E44_common_target/code")
import e44_lib as L  # noqa: E402

torch.set_num_threads(1)
SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
SIZES = [25, 50]
E44_CELLS = Path("E44_common_target/results/cells")
CELLS = Path("E46_low_data_basis/results/cells")
CODE = ["E46_low_data_basis/code/run_e46.py", "E46_low_data_basis/code/analyze_e46.py"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def require_prereg(commit):
    full = git("rev-parse", commit)
    subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], check=True)
    assert "### E46." in git("show", f"{full}:EXPERIMENTS.md"), "no E46 block at prereg commit"
    assert git("diff", "--name-only", full, "HEAD", "--", *CODE) == "", "code changed since prereg"
    assert git("status", "--porcelain", "--untracked-files=no", "--", *CODE) == "", "code not committed"
    return full


# ---------------- arms of part B ----------------
class PrescribedSinCos(nn.Module):
    def forward(self, x):
        p = x[..., :4] / 512
        th = x[..., 4]
        return torch.cat([p, torch.sin(th).unsqueeze(-1), torch.cos(th).unsqueeze(-1)], -1)


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


# ---------------- checks and progress, wrapped from outside ----------------
_train = L._e44_train
_checks = []


def _checked_train(mdl, tl, vl, epochs):
    if L._E44["target"] != "external":
        return _train(mdl, tl, vl, epochs)
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
    print(f"{_prog['tag']} stage {st} epoch {(_prog['ep'] - 1) % L.EPOCHS + 1:2d}/{L.EPOCHS} "
          f"{time.time() - _prog['t0']:6.0f} s", flush=True)
    return v


L.val_loss = _progress


def stage2_only(enc, d, tl, vl, seed):
    """Stage 2 of E44 run_arm for a fixed encoder of output width d."""
    torch.manual_seed(seed)
    L.set_target("external")
    try:
        mdl = L.WorldModel(enc, d)
        if d != 5:
            h = mdl.pr.net[-1].in_features
            mdl.pr.net[-1] = nn.Linear(h, 5)
        _checks.clear()
        hist = L._e44_train(mdl, tl, vl, L.EPOCHS)
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
    assert L.EPOCHS == 30
    CELLS.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        f = CELLS / f"seed_{s}.json"
        if f.exists():
            print(f"seed {s}: exists, skipped", flush=True)
            continue
        rec = {"seed": s, "prereg": full, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "lowdata": {}, "basis": {}}
        # part A
        for n in SIZES:
            L.EPISODES = n
            rec["lowdata"][str(n)] = {}
            for arm in ("prescribed", "free_scaled"):
                tl, vl = L.make_data(s)
                assert len(tl) >= 5, f"too few batches at {n} episodes: {len(tl)}"
                _prog.update(tag=f"seed {s} n{n} {arm}", ep=0, t0=time.time())
                _checks.clear()
                r = L.run_arm(arm, tl, vl, s, stage2=True)
                r["encoder_unchanged_in_stage2"] = len(_checks) == 1 and _checks[0]
                r["batches_per_epoch"] = len(tl)
                rec["lowdata"][str(n)][arm] = r
        # part B
        L.EPISODES = 200
        e44 = json.loads((E44_CELLS / f"seed_{s}.json").read_text())
        tl, vl = L.make_data(s)
        for arm in ("prescribed_gate", "prescribed_sincos", "prescribed_std"):
            _prog.update(tag=f"seed {s} n200 {arm}", ep=0, t0=time.time())
            if arm == "prescribed_gate":
                enc, d = L.PrescribedEncoder(5), 5
            elif arm == "prescribed_sincos":
                enc, d = PrescribedSinCos(), 6
            else:
                mu, sd = train_stats(tl)
                enc, d = PrescribedStd(mu, sd), 5
            r = stage2_only(enc, d, tl, vl, s)
            if arm == "prescribed_gate":
                r["equals_e44"] = r["s2_hist"] == e44["arms"]["prescribed"]["s2_hist"]
            rec["basis"][arm] = r
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
