#!/usr/bin/env python3
"""Create E39's private copy of the E32 training library, with checkpointing.

e32_lib.py is shared by E32 and E38, whose results are already in the registries.
It is not edited. This produces e39_lib.py alongside it in E39's code directory
and applies three substitutions, each guarded by assert count == 1.

Added behaviour: when ckpt_prefix is given, run_subepoch writes
  <prefix>_enc_at_freeze.pt   encoder state at the moment of freezing
  <prefix>_model_final.pt     full model state after all epochs
Without ckpt_prefix the function behaves exactly as in e32_lib.

Usage: python patch_e39_lib.py
"""
import shutil, os

SRC = "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code/e32_lib.py"
DST = "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code/e39_lib.py"

os.makedirs(os.path.dirname(DST), exist_ok=True)
shutil.copyfile(SRC, DST)
src = open(DST).read()

EDITS = [
    ("E39-1 signature",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free'):",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free', ckpt_prefix=None):"),

    ("E39-2 save encoder at freeze",
     "        for p in mdl.enc.parameters(): p.requires_grad = False",
     "        for p in mdl.enc.parameters(): p.requires_grad = False\n"
     "        if ckpt_prefix is not None:\n"
     "            torch.save(mdl.enc.state_dict(), ckpt_prefix + \"_enc_at_freeze.pt\")"),

    ("E39-3 save final model",
     "    best = min(h['vp'] for h in hist)",
     "    if ckpt_prefix is not None:\n"
     "        torch.save(mdl.state_dict(), ckpt_prefix + \"_model_final.pt\")\n"
     "    best = min(h['vp'] for h in hist)"),
]

for name, old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"{name}: expected 1 match, found {n}"
    src = src.replace(old, new)
    print(f"ok {name}")

open(DST, "w").write(src)
print(f"written {DST}")
