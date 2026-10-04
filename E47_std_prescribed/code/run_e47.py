#!/usr/bin/env python3
# marker: E47 v1
"""E47. Run from the repository root:
  OMP_NUM_THREADS=1 python3 -u E47_std_prescribed/code/run_e47.py --prereg <sha> --seeds ...

prescribed_std (fixed, five prescribed features standardised on the training
split of the seed and size) against free_scaled, on the common target, 90
epochs per stage, at 25, 50 and 200 episodes.
- free_scaled at 25 and 50: both stages, as E45 (run_arm with epochs=90).
- free_scaled at 200: taken from E45 cells (same seeds, data, code path).
- prescribed_std at 25, 50, 200: stage 2 only (encoder fixed), as E46 part B.
- gate, seed 42 only: plain prescribed stage 2 at 200 must equal E45 prescribed
  s2_hist bit for bit.
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
EPOCHS = 90
E45_CELLS = Path("E45_long_training/results/cells")
CELLS = Path("E47_std_prescribed/results/cells")
CODE = ["E47_std_prescribed/code/run_e47.py", "E47_std_prescribed/code/analyze_e47.py"]
L.EPOCHS = EPOCHS  # used by stage2_only and the progress line; run_arm gets epochs explicitly


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def require_prereg(commit):
    full = git("rev-parse", commit)
    subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], check=True)
    assert "### E47." in git("show", f"{full}:EXPERIMENTS.md"), "no E47 block at prereg commit"
    assert git("diff", "--name-only", full, "HEAD", "--", *CODE) == "", "code changed since prereg"
    assert git("status", "--porcelain", "--untracked-files=no", "--", *CODE) == "", "code not committed"
    return full


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
    CELLS.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        f = CELLS / f"seed_{s}.json"
        if f.exists():
            print(f"seed {s}: exists, skipped", flush=True)
            continue
        rec = {"seed": s, "prereg": full, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "epochs": EPOCHS, "sizes": {}}
        for n in (25, 50, 200):
            L.EPISODES = n
            rec["sizes"][str(n)] = {}
            if n != 200:
                tl, vl = L.make_data(s)
                assert len(tl) >= 5, (n, len(tl))
                _prog.update(tag=f"seed {s} n{n} free_scaled", ep=0, t0=time.time())
                _checks.clear()
                r = L.run_arm("free_scaled", tl, vl, s, stage2=True, epochs=EPOCHS)
                r["encoder_unchanged_in_stage2"] = len(_checks) == 1 and _checks[0]
                rec["sizes"][str(n)]["free_scaled"] = r
            tl, vl = L.make_data(s)
            mu, sd = train_stats(tl)
            _prog.update(tag=f"seed {s} n{n} prescribed_std", ep=0, t0=time.time())
            r = stage2_only(PrescribedStd(mu, sd), tl, vl, s)
            r["mu"], r["sd"] = mu.tolist(), sd.tolist()
            rec["sizes"][str(n)]["prescribed_std"] = r
        if s == 42:
            L.EPISODES = 200
            tl, vl = L.make_data(s)
            _prog.update(tag="seed 42 n200 prescribed_gate", ep=0, t0=time.time())
            r = stage2_only(L.PrescribedEncoder(5), tl, vl, s)
            e45 = json.loads((E45_CELLS / "seed_42.json").read_text())
            rec["gate_equals_e45"] = r["s2_hist"] == e45["arms"]["prescribed"]["s2_hist"]
            print(f"seed 42 gate equals E45: {rec['gate_equals_e45']}", flush=True)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
