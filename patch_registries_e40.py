#!/usr/bin/env python3
"""Record E40: Ф64, the split verdict on Г26, and the new Г27.

Four substitutions, each guarded by assert count == 1:
  EV-1  Ф64 inserted after the Ф63 block
  EV-2  Г26 status replaced with the E40 verdict; Г27 appended after it
  EX-1  E40 marked complete in the experiment list
  EX-2  E40 row in the July experiment table

Usage: python patch_registries_e40.py
"""
import io

EV = "/mnt/d/coordinate-stability/EVIDENCE.md"
EX = "/mnt/d/coordinate-stability/EXPERIMENTS.md"

F, G = "\u0424", "\u0413"

EV_1_OLD = ("- Artefact: E39_subepoch_freeze_micro/code/analyze_representation.py, "
            "E39_subepoch_freeze_micro/analysis/repr_seed_*.json\n- E39\n")

EV_1_NEW = EV_1_OLD + f"""
**{F}64. What a fixed initialisation carries does not predict how good a coordinate system it is (E40)**
- Design: the data seed is held at 42 and only the encoder initialisation varies (10 initialisations). The encoder is frozen at step 0, so it never trains. The DataLoader stream is restored after construction, so batch order is identical across initialisations and only the parameters differ. This is the first measurement in the line that separates initialisation from data sample.
- best_vp by initialisation 1 to 10: 0.00314, 0.00399, 0.00386, 0.00483, 0.00189, 0.00273, 0.00893, 0.00354, 0.00122, 0.00352. R2_readout over the same: 0.2385, 0.3377, 0.4094, 0.4716, 0.2494, 0.5156, 0.4115, 0.5137, 0.4972, 0.3886.
- **corr(best_vp, R2_readout) = +0.060** over a two-fold range of R2_readout (0.239 to 0.516). Informativeness of a frozen initialisation does not predict the downstream result. The inverse ordering visible in the five E39 seeds was an appearance produced by initialisation and data sample varying together.
- best_vp nevertheless spreads **7.31x** across initialisations (0.00122 to 0.00893) at a fixed data sample. Fixed bases differ strongly from each other; what separates them is not what they carry.
- The prescribed encoder scores 0.00261 on the same data, inside the range of the random ones, with 2 of 10 initialisations better than it. The prescribed coordinate readout is not distinguished among frozen random bases (strengthens {F}31, {F}60).
- corr(best_vp, eff_rank) = -0.287, in the direction of higher rank being better, but not significant at n=10. Recorded as a lead, not a claim.
- Environment: Push-T real gym-pusht, data seed 42, EP=15/NEP=200, freeze at step 0, e40_lib (init_seed separable), local CPU.
- Caveats: one data sample only, so the 7.31x spread is within-sample across initialisations and its dependence on the sample is unmeasured. n=10 supports the null on R2_readout but not a claim about eff_rank.
- Artefact: E40_init_sweep/ (README + code + results/sweep.json)
- E40
"""

EV_2_OLD = f"""- Status: OPEN, untested
- E39, E40 (planned)
"""

EV_2_NEW = f"""- Status: SPLIT by E40 ({F}64). Confirmed in the negative part: informativeness does not decide, corr(best_vp, R2_readout) = +0.060 at fixed data. Refuted in the positive part: immobility alone does not decide either, since frozen initialisations spread 7.31x among themselves. What distinguishes one fixed basis from another is open and is carried forward as {G}27.
- E39, E40

**{G}27. Something other than linear informativeness distinguishes one frozen basis from another**
- At a fixed data sample, frozen random initialisations spread 7.31x in best_vp while their linear readout of the true state is uncorrelated with that spread ({F}64). The quantity that orders them is not identified.
- The available lead is the effective rank of the representation, corr -0.287 at n=10, in the direction of higher rank being better. eff_rank also varies by initialisation in E39 (1.61 to 2.70 of a maximum 3) and is close to flat across the opening window ({F}63), so it is a property of the initialisation rather than of training.
- Falsifier: a sweep with enough initialisations to settle whether eff_rank, or any other cheap property of the untrained encoder, predicts best_vp. If none does, the spread has to be attributed to the interaction with the downstream module rather than to the encoder alone.
- Test: E41. Initialisations at two or three data seeds, n large enough for a rank correlation to mean something, candidate predictors computed before training.
- Status: OPEN, untested
- E40
"""

EX_1_OLD = (f"- **E40**: initialisation sweep at a fixed data seed, frozen at step 0 (PLANNED, {G}26).")
EX_1_NEW = (f"- **E40**: initialisation sweep at a fixed data seed, frozen at step 0 "
            f"(COMPLETE 2026-09-11, {F}64, {G}26 split, {G}27).")

EX_2_OLD = ("| E39 | Sub-epoch freeze micro-grid | Push-T gym | gym | 5 | 15 | 200 | "
            "JEPA initial-collapse confound closed: no recovery segment in any seed, 3/5 monotone, "
            f"dips at different steps ({F}61); action_space seeding defect found ({F}62) |")
EX_2_NEW = (EX_2_OLD + "\n"
            "| E40 | Initialisation sweep, fixed data seed | Push-T gym | gym | 10 inits | 15 | 200 | "
            "corr(best_vp, R2_readout) = +0.060; best_vp spreads 7.31x across initialisations; "
            f"prescribed inside the range ({F}64) |")

EDITS = {
    EV: [(f"EV-1 {F}64", EV_1_OLD, EV_1_NEW),
         (f"EV-2 {G}26 verdict and {G}27", EV_2_OLD, EV_2_NEW)],
    EX: [("EX-1 E40 complete", EX_1_OLD, EX_1_NEW),
         ("EX-2 E40 row", EX_2_OLD, EX_2_NEW)],
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
