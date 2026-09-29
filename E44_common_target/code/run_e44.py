#!/usr/bin/env python3
# marker: E44 runner v2
"""E44 campaign runner.

python3 E44_common_target/code/run_e44.py --prereg <commit> --seeds 42 123 777

Refuses to run unless <commit> is an ancestor of HEAD, holds exactly one
"### E44." block in EXPERIMENTS.md, touches e44_lib.py, run_e44.py and
analyze_e44.py, and the working-tree copies of those files equal the commit.
Writes E44_common_target/results/cells/seed_<s>.json per seed; a seed whose
file exists is skipped. Prints no loss values.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "E44_common_target/code")
import e44_lib as L

ARMS = ["prescribed", "prescribed_rotated", "free_raw", "free_scaled"]
SEEDS = [42, 123, 777, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
CODE = ["E44_common_target/code/e44_lib.py",
        "E44_common_target/code/run_e44.py",
        "E44_common_target/code/analyze_e44.py"]
CELLS = Path("E44_common_target/results/cells")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def require_prereg(commit):
    r = git("rev-parse", "--verify", commit + "^{commit}")
    assert r.returncode == 0, f"unknown commit {commit}"
    full = r.stdout.strip()
    assert git("merge-base", "--is-ancestor", full, "HEAD").returncode == 0, \
        "pre-registration commit is not an ancestor of HEAD"
    ex = git("show", f"{full}:EXPERIMENTS.md")
    assert ex.returncode == 0 and ex.stdout.count("### E44.") == 1, \
        "E44 block missing from EXPERIMENTS.md in the pre-registration commit"
    names = git("show", "--name-only", "--format=", full).stdout.split()
    for c in CODE:
        assert c in names, f"{c} is not in the pre-registration commit"
    assert git("diff", "--quiet", full, "--", *CODE).returncode == 0, \
        "code differs from the pre-registration commit"
    return full


# Stage-2 gate: the encoder state must be tensor-equal before and after
# stage 2. Wraps the library's training loop from outside; e44_lib.py is not
# modified.
_train = L._e44_train
_checks = []


def _checked_train(mdl, tl, vl, epochs):
    if L._E44["target"] != "external":
        return _train(mdl, tl, vl, epochs)
    before = {k: v.detach().clone() for k, v in mdl.enc.state_dict().items()}
    hist = _train(mdl, tl, vl, epochs)
    after = mdl.enc.state_dict()
    _checks.append(set(before) == set(after) and
                   all(torch.equal(before[k], after[k]) for k in before))
    return hist


L._e44_train = _checked_train

# Progress: one line per epoch (seed, arm, stage, epoch, elapsed). Wraps
# val_loss from outside; prints no loss values.
_val_loss = L.val_loss
_prog = {"seed": None, "arm": None, "epoch": 0, "t0": 0.0}


def _progress_val_loss(mdl, vl):
    v = _val_loss(mdl, vl)
    _prog["epoch"] += 1
    stage = 1 if L._E44["target"] == "self" else 2
    ep = (_prog["epoch"] - 1) % L.EPOCHS + 1
    print(f"seed {_prog['seed']} {_prog['arm']} stage {stage} epoch {ep:2d}/{L.EPOCHS} "
          f"{time.time() - _prog['t0']:6.0f} s", flush=True)
    return v


L.val_loss = _progress_val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for s in a.seeds:
        assert s in SEEDS, f"seed {s} is not registered"
    full = require_prereg(a.prereg)
    CELLS.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        f = CELLS / f"seed_{s}.json"
        if f.exists():
            print(f"seed {s}: exists, skipped", flush=True)
            continue
        rec = {"seed": s, "prereg": full, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "arms": {}}
        for arm in ARMS:
            t0 = time.time()
            _prog.update(seed=s, arm=arm, epoch=0, t0=t0)
            tl, vl = L.make_data(s)
            _checks.clear()
            r = L.run_arm(arm, tl, vl, s, stage2=True)
            r["encoder_unchanged_in_stage2"] = len(_checks) == 1 and _checks[0]
            r["cell_seconds"] = round(time.time() - t0, 1)
            rec["arms"][arm] = r
            print(f"seed {s} {arm}: {r['cell_seconds']} s", flush=True)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.rename(f)
        print(f"seed {s}: written", flush=True)


if __name__ == "__main__":
    main()
