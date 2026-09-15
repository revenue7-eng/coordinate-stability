#!/usr/bin/env python3
"""E42 candidate properties, computed on the untrained encoder.

Closed list fixed by the pre-registration in EXPERIMENTS.md (E42 block, amended
before the run in a97f59c and d9dee43). Nothing in this file is chosen after
seeing a result.

All four candidates and the control read the same states: the full validation
split returned by E39's `val_states(data_seed)`, which is the set behind {F}64
and {F}67. `eff_rank` and `r2_readout` are imported from E39's
`analyze_representation` rather than reimplemented, so the values produced here
are the same quantities as the 1.5380 and 1.5755 recorded in {F}67. A copy would
have been a second definition free to drift.

`val_states` flattens the window axis away. Candidate 4 needs it, so the
flattened tensor is folded back to (Nv, T, 5) rather than the split being
rebuilt: folding cannot disagree with `val_states`, a second copy of the split
logic could.

The encoder is applied twice, once to the flat (N, 5) tensor and once to the
folded (Nv, T, 5) one. The flat pass is the path E40 used for {F}64 and {F}67
and is what candidates 1 to 3 read, so those numbers stay comparable to the
recorded ones without relying on batch shape leaving the arithmetic untouched.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE,
           "/mnt/d/coordinate-stability/E39_subepoch_freeze_micro/code",
           "/mnt/d/coordinate-stability/E32_subepoch_freeze_real/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_representation import eff_rank, r2_readout  # noqa: E402

T_WINDOW = 5       # DS(eps, H=3) stores H+2 = 5 consecutive states per item
OUT_DIM = 3        # FE and PE both emit 3 dimensions


def fold_windows(S, T=T_WINDOW):
    """(N, 5) as returned by val_states -> (Nv, T, 5), its original windows."""
    assert S.shape[0] % T == 0, (tuple(S.shape), T)
    return S.reshape(-1, T, S.shape[-1])


def _cov_eigenvalues(rep):
    """Eigenvalues of the representation covariance, the matrix eff_rank reads."""
    w = np.linalg.eigvalsh(np.cov(rep.T))
    return np.clip(w, 0.0, None)


def cond_number(rep):
    """Candidate 3: largest over smallest eigenvalue of that same covariance.

    Sensitive to the worst-conditioned direction rather than to how evenly
    variance is spread, which is what makes it a hypothesis distinct from
    eff_rank rather than a restatement of it.
    """
    w = _cov_eigenvalues(rep)
    if w.min() <= 0.0:
        return float("inf")
    return float(w.max() / w.min())


def smoothness(z3):
    """Candidate 4: mean(||z_{t+1} - z_t||) / sqrt(mean(||z_t||^2)).

    z3 is (Nv, T, OUT_DIM). The pairs are within-window, so the DataLoader
    shuffle is irrelevant to them. Windows are built at stride 1 and therefore
    overlap: interior states enter both means up to T times, the observations
    behind this candidate are not independent, and the jackknife reported for it
    says so.
    """
    steps = np.linalg.norm(z3[:, 1:] - z3[:, :-1], axis=-1)
    return float(steps.mean()) / rms_norm(z3)


def rms_norm(z3):
    """Control, outside the multiplicity correction: scale of the encoder output.

    It is the denominator of candidate 4, which is what makes that candidate
    scale-free. Reported as a sanity check on whether SIGReg equalises scale
    across initialisations. Never counted as a hit.
    """
    return float(np.sqrt((z3 ** 2).sum(-1).mean()))


@torch.no_grad()
def candidates(enc, S, target):
    """Four candidates and the control for one untrained encoder.

    enc     an encoder in eval mode, loaded from its _enc_at_freeze checkpoint
    S       (N, 5) validation states from val_states(data_seed)
    target  (N, 3) true coordinates, PE()(S), as E40 built them
    """
    rep = enc(S).numpy()
    z3 = enc(fold_windows(S)).numpy()
    assert rep.shape == (S.shape[0], OUT_DIM), (rep.shape, S.shape)
    assert z3.shape[0] * z3.shape[1] == rep.shape[0], (z3.shape, rep.shape)
    return {"eff_rank": float(eff_rank(rep)),
            "r2_readout": float(r2_readout(rep, target)),
            "cond_number": cond_number(rep),
            "smoothness": smoothness(z3),
            "rms_norm": rms_norm(z3)}


CANDIDATE_KEYS = ("eff_rank", "r2_readout", "cond_number", "smoothness")
CONTROL_KEYS = ("rms_norm",)
