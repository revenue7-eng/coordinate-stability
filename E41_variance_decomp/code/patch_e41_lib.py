#!/usr/bin/env python3
"""Create E41's copy of the library, with encoder and head initialisation separable.

e40_lib.run_subepoch reseeds the global stream once from `init_seed` immediately
before the model is built, so one argument draws the parameters of FE, AE and PR
together. With the encoder frozen at step 0 the head (AE + PR) is the only part
that trains, so the spread measured in E40 mixes encoder initialisation with head
initialisation and cannot be attributed to the encoder alone.

This splits the argument in two: `enc_seed` is applied before the encoder is
constructed, `head_seed` before M() draws AE and PR. With head_seed=None the
stream position is identical to e40_lib with init_seed=enc_seed, so E40 cells
must reproduce exactly. That equality is the acceptance test for this patch.
"""
import os, shutil

SRC = "/mnt/d/coordinate-stability/E40_init_sweep/code/e40_lib.py"
DST = "/mnt/d/coordinate-stability/E41_variance_decomp/code/e41_lib.py"

os.makedirs(os.path.dirname(DST), exist_ok=True)
shutil.copyfile(SRC, DST)
src = open(DST).read()

EDITS = [
    ("E41-1 signature",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free', ckpt_prefix=None, init_seed=None):",
     "def run_subepoch(eps, seed, epochs, freeze_frac, mode='free', ckpt_prefix=None, enc_seed=None, head_seed=None):"),

    ("E41-2 separable encoder and head init",
     "    if init_seed is not None:\n"
     "        torch.manual_seed(init_seed)\n"
     "    enc = PE() if mode == 'prescribed' else FE()\n"
     "    mdl = M(enc, AE(), PR(), SIGReg())\n"
     "    if init_seed is not None:\n"
     "        torch.manual_seed(seed)",
     "    if enc_seed is not None:\n"
     "        torch.manual_seed(enc_seed)\n"
     "    enc = PE() if mode == 'prescribed' else FE()\n"
     "    if head_seed is not None:\n"
     "        torch.manual_seed(head_seed)\n"
     "    mdl = M(enc, AE(), PR(), SIGReg())\n"
     "    if enc_seed is not None or head_seed is not None:\n"
     "        torch.manual_seed(seed)"),
]

for name, old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"{name}: expected 1 match, found {n}"
    src = src.replace(old, new)
    print(f"ok {name}")

assert src.count("init_seed") == 0, "init_seed is still referenced"
open(DST, "w").write(src)
print(f"written {DST}")
