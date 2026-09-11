#!/usr/bin/env python3
"""Create E40's copy of the library, with the model initialisation separable
from the data seed.

run_subepoch seeds one global stream from `seed`, and the model parameters are
drawn from it, so initialisation cannot be varied without varying the data.
This adds an optional init_seed: the stream is reseeded just before the model
is built and restored to `seed` immediately after, so that the DataLoader
shuffling order is identical across initialisations and only the parameters
differ.

With init_seed=None the function behaves exactly as in e39_lib.

Note: E40 runs are comparable among themselves, not against E39 step 0, because
restoring the stream after construction leaves the loader order at a different
point than the untouched path does.

Usage: python patch_e40_lib.py
"""
import shutil, os

SRC = "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code/e39_lib.py"
DST = "/mnt/d/coordinate-stability/E40_init_sweep/code/e40_lib.py"

os.makedirs(os.path.dirname(DST), exist_ok=True)
shutil.copyfile(SRC, DST)
src = open(DST).read()

EDITS = [
    ("E40-1 signature",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free', ckpt_prefix=None):",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free', ckpt_prefix=None, init_seed=None):"),

    ("E40-2 separable init",
     "    enc = PE() if mode == 'prescribed' else FE()\n"
     "    mdl = M(enc, AE(), PR(), SIGReg())",
     "    if init_seed is not None:\n"
     "        torch.manual_seed(init_seed)\n"
     "    enc = PE() if mode == 'prescribed' else FE()\n"
     "    mdl = M(enc, AE(), PR(), SIGReg())\n"
     "    if init_seed is not None:\n"
     "        torch.manual_seed(seed)"),
]

for name, old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"{name}: expected 1 match, found {n}"
    src = src.replace(old, new)
    print(f"ok {name}")

open(DST, "w").write(src)
print(f"written {DST}")
