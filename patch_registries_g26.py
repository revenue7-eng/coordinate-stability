#!/usr/bin/env python3
"""Record the E39 representation analysis: Ф63 (observation) and Г26 (hypothesis).

Three substitutions, each guarded by assert count == 1:
  EV-1  Ф63 inserted after the Ф62 block
  EV-2  Г26 inserted after Г25, at the end of the HYPOTHESES section
  EX-1  E40 registered as planned; the free slot moves to E41+

Usage: python patch_registries_g26.py
"""
import io

EV = "/mnt/d/coordinate-stability/EVIDENCE.md"
EX = "/mnt/d/coordinate-stability/EXPERIMENTS.md"

F, G = "\u0424", "\u0413"

EV_1_OLD = ("- Artefact: E39_subepoch_freeze_micro/code/patch_e39_seedfix.py; "
            "E39_subepoch_freeze_micro/prefix_bug/ (results of the defective collection, "
            "retained as evidence)\n- E39\n")

EV_1_NEW = EV_1_OLD + f"""
**{F}63. Inside the opening window the representation gains linear information about the true state while the downstream result gets worse (E39)**
- Probe: for each grid point, the encoder checkpoint taken at the moment of freezing is run over that run's own validation states, and three quantities are computed. R2_readout is the R2 of a least-squares linear map from the 3-dimensional representation to the true scaled coordinates. Procrustes is the disparity against the previous grid point after centering, scaling and optimal rotation. eff_rank is the exponentiated entropy of the representation covariance spectrum (maximum 3).
- R2_readout rises monotonically over steps 0 to 8 in **5/5 seeds**, including the two whose best_vp is not monotone: 0.336 to 0.381 (seed 42), 0.424 to 0.452 (123), 0.337 to 0.367 (777), 0.311 to 0.355 (2024), 0.444 to 0.472 (7).
- Over the same steps best_vp rises as well ({F}61). Information and error increase together, so what improves in the representation is not what determines the result.
- At step 0 an untrained random encoder already reaches R2_readout 0.31 to 0.44 and performs within one order of magnitude of the prescribed encoder, which is a parameter-free readout of the true state (PE is x[...,2:5]*sc) and therefore scores 1 by construction ({F}31, {F}60).
- Procrustes disparity from the previous point is small over steps 1 to 4 (0.0004 to 0.0021) and 2 to 4 times larger at steps 6 and 8 in 5/5 seeds: the basis accelerates.
- eff_rank is set by initialisation (1.61, 1.79, 2.09, 2.09, 2.70) and is close to flat across steps. The only visible decline is seed 2024, 2.70 to 2.54, which is also the only seed with a pronounced dip in best_vp. One case, no claim.
- The prediction recorded before the computation was that R2_readout would hold constant, on the reading that only the basis moves while content is preserved. It is refuted: the content changes too, and monotonically.
- Sanity: the prescribed encoder scores exactly 1.0000 in the same pipeline.
- Environment: no training, forward passes only; validation split reproduced with the run's own DS and random_split generator seed; e39_lib, local CPU.
- Artefact: E39_subepoch_freeze_micro/code/analyze_representation.py, E39_subepoch_freeze_micro/analysis/repr_seed_*.json
- E39
"""

EV_2_OLD = ("- Status: OPEN, untested. Support in this environment withdrawn; "
            "the assigned test is on hold pending a metric with established sensitivity\n"
            "\n---\n")

EV_2_NEW = (EV_2_OLD.replace("\n---\n", "") + f"""
**{G}26. What the result depends on is the immobility of the representation, not its informativeness**
- Inside the opening window the two move in opposite directions: linear information about the true state rises monotonically in 5/5 seeds while best_vp gets worse ({F}63). A random untrained encoder carrying roughly a third of the linearly extractable information performs on a par with a parameter-free readout carrying all of it ({F}31, {F}60, {F}63).
- The hypothesis is about movement rather than content: damage is done by the representation continuing to change under a downstream module that is adapting to it, and neither by a poor choice of axes nor by loss of information.
- Falsifier: a controlled comparison in which best_vp tracks the initialisation's R2_readout. Across the five available points the two are, if anything, inversely ordered (seed 7 has the highest R2_readout at 0.444 and the worst best_vp at 0.00843), but those five points confound initialisation with data sample and are an observation, not a test.
- Test: E40. Several encoder initialisations at a fixed data seed, frozen at step 0, best_vp measured against the initialisation's R2_readout and eff_rank. This is the first measurement in the line that separates initialisation from sample.
- Status: OPEN, untested
- E39, E40 (planned)

---
""")

EX_1_OLD = "- **E40+**: free. The nearest candidate is ECA / epiplexity (\u041317)."
EX_1_NEW = (f"- **E40**: initialisation sweep at a fixed data seed, frozen at step 0 (PLANNED, {G}26).\n"
            "- **E41+**: free. The nearest candidate is ECA / epiplexity (\u041317).")

EDITS = {
    EV: [(f"EV-1 {F}63", EV_1_OLD, EV_1_NEW),
         (f"EV-2 {G}26", EV_2_OLD, EV_2_NEW)],
    EX: [("EX-1 E40 planned", EX_1_OLD, EX_1_NEW)],
}

for path, edits in EDITS.items():
    src = io.open(path, encoding="utf-8").read()
    for name, old, new in edits:
        n = src.count(old)
        assert n == 1, f"{name}: expected 1 match, found {n}"
        src = src.replace(old, new)
        print(f"ok {name}")
    io.open(path, "w", encoding="utf-8", newline="").write(src)
    print(f"written {path}")
