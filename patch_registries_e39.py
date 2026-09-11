#!/usr/bin/env python3
"""Apply E39 findings to EVIDENCE.md and EXPERIMENTS.md.

Six substitutions, each guarded by assert count == 1:
  EV-1  header date
  EV-2  Ф61 and Ф62 inserted after the Ф60 block
  EV-3  Ф46 caveats: pointer to Ф62
  EV-4  Ф60 caveats: pointer to Ф61 and Ф62
  EX-1  E39 row in the July experiment table
  EX-2  free-slot note moved from E39+ to E40+

Usage: python patch_registries_e39.py
"""
import io

EV = "/mnt/d/coordinate-stability/EVIDENCE.md"
EX = "/mnt/d/coordinate-stability/EXPERIMENTS.md"

# ---------------------------------------------------------------- EVIDENCE

EV_1_OLD = "Last updated: 25 August 2026 (\u042457, \u041d1/\u041d2 refuted, \u041325 rewritten)"
EV_1_NEW = ("Last updated: 11 September 2026 (\u042461/\u042462 added, \u042446/\u042460 caveats amended; "
            "earlier: \u042457, \u041d1/\u041d2 refuted, \u041325 rewritten)")

EV_2_OLD = ("- Artefact: E38_subepoch_freeze_full/ (README + code + per-seed results)\n"
            "- E38\n")

EV_2_NEW = EV_2_OLD + """
**\u042461. The JEPA initial-collapse confound does not explain the sub-epoch damage (E39)**
- Question: inside epoch 1, is the measured quantity coordinate-basis drift, or the collapse-then-recover transient that T-JEPA/I-JEPA report in the first iterations? E38's finest point (f=0.05) is optimizer step 8 of 160, so the whole opening window sat below its resolution.
- Design: grid specified in optimizer steps {0,1,2,3,4,6,8} rather than fractions, f=(step+0.5)/n_batches so that floor(f*n_batches) lands on the intended step regardless of float representation. n_batches=160 (measured), EP=15/NEP=200, 5 seeds {42,123,777,2024,7}, real gym-pusht, local CPU.
- Discriminating prediction: collapse-then-recover requires a recovery segment, a rise to a peak followed by a sustained fall. The encoder is frozen at step f and stays frozen for all remaining epochs, so it cannot recover from a dip, and such a segment would be visible. **No seed shows one.**
- best_vp over steps 0,1,2,3,4,6,8: seed 42 0.00330/0.00333/0.00339/0.00347/0.00357/0.00388/0.00430; seed 123 0.00181/0.00180/0.00184/0.00185/0.00188/0.00200/0.00213; seed 777 0.00282/0.00312/0.00340/0.00374/0.00401/0.00486/0.00587; seed 2024 0.00101/0.00103/0.00110/0.00105/0.00110/0.00113/0.00126; seed 7 0.00843/0.00846/0.00853/0.00858/0.00859/0.00885/0.00923.
- Strictly monotone in **3/5** seeds (42, 777, 7). Two seeds show a single dip at different steps (123 at step 1, -0.6%; 2024 at step 3, -4.5%), each followed by continued rise. Do NOT state 5/5 for this window.
- The dips are not measurement noise: run_subepoch is deterministic given (eps, seed), verified by two identical runs in one process (0.003017362545391447 twice). Each point is an exact value, so a dip is a property of the pair (sample, step).
- Shape is invariant to the sampling defect of \u042462: the same grid on two independent samples for seed 42 (before and after the fix) gives window ratio 1.23x and 1.30x, monotone in both.
- Supported by \u042445: there is no recovery after epoch 1 either (freeze@1 to unfrozen = 1.3x).
- Consequence: the open confound recorded in \u042460's caveats is closed. The E30/E31/E32/E38 line measures damage that accumulates and persists, not a transient that resolves.
- Caveats: the window ratio varies strongly by sample (1.10x to 2.08x), and the between-sample spread at step 0 is 8.3x (0.00101 to 0.00843). Nothing below one optimizer step is resolved. The absence of a recovery segment is established for this architecture (predictor with stop-grad plus SIGReg, no EMA target encoder), not for JEPA variants in general.
- Artefact: E39_subepoch_freeze_micro/ (README + code + per-seed results + checkpoints)
- E39

**\u042462. collect_gym_data in e32_lib does not reproduce for a given seed (action space unseeded)**
- The function seeds its own Generator and uses it for env.reset, the branch draw and the noise draw, but `env.action_space.sample()` draws from the action space's own generator, which gym.make initialises from system entropy. That branch fires on the first step of every episode and in roughly 30% of later steps, so about a third of all recorded actions came from an unseeded source.
- Probe: two calls with seed=42 in one process give different SHA256 of the pickled episodes, and a second process gives two more distinct hashes (4 of 4 different). After adding `env.action_space.seed(int(rng.integers(0, 100000)))` all four agree (859c6a33fc96a239).
- Magnitude: at nominal seed 42 and f=0.00, three processes produced best_vp 0.00302, 0.00327 (E38 as shipped) and 0.00332, a spread of about 10%.
- Scope, affected: absolute numbers from E32 and E38, and any cross-run comparison of them. The label "5 seeds" in \u042446 and \u042460 means five independent samples whose labels do not identify them, not five controlled repetitions of one condition; the reported spread is between-sample.
- Scope, NOT affected: every shape verdict and within-run comparison. A run collects its data once and all grid points of that run share it, so the curve shape is measured on one fixed sample.
- Decision: E38 is not rerun. The load-bearing claims are shape claims and are unaffected, the absolute anchors already carry a do-not-compare caveat, and E39 demonstrated shape invariance across the defect directly. The fix lives in E39_subepoch_freeze_micro/code/e39_lib.py; e32_lib.py is left untouched so that E32/E38 artefacts remain reproducible as recorded.
- Also refutes the inherited reasoning that thread count cannot affect a result "because the seed is deterministic". The premise was false; measured, threads change the ninth decimal only (0.003017362545391447 at 4 threads vs 0.003017361605899376 at 1), so the conclusion happened to hold.
- Artefact: E39_subepoch_freeze_micro/code/patch_e39_seedfix.py; E39_subepoch_freeze_micro/prefix_bug/ (results of the defective collection, retained as evidence)
- E39
"""

EV_3_OLD = "A full-fidelity rerun (EP=15, NEP=200) is a one-line change."
EV_3_NEW = (EV_3_OLD + " The 5 seeds are five independent samples rather than controlled "
            "repetitions, and the absolute numbers do not reproduce across runs (\u042462); "
            "the shape verdict is unaffected.")

EV_4_OLD = "is NOT settled by this experiment."
EV_4_NEW = ("is NOT settled by this experiment, and is closed separately by E39 (\u042461). "
            "The 5 seeds are five independent samples rather than controlled repetitions (\u042462).")

# ------------------------------------------------------------- EXPERIMENTS

EX_1_OLD = ("| E38 | Sub-epoch freeze full budget | Push-T gym | gym | 5 | 15 | 200 | "
            "SLOPE on [0.00,0.40] R\u00b2=0.880 5/5 monotone; onset 5.8\u201317.1\u00d7 (\u042460); "
            "\u042446 onset reading revised |")
EX_1_NEW = (EX_1_OLD + "\n"
            "| E39 | Sub-epoch freeze micro-grid | Push-T gym | gym | 5 | 15 | 200 | "
            "JEPA initial-collapse confound closed: no recovery segment in any seed, 3/5 monotone, "
            "dips at different steps (\u042461); action_space seeding defect found (\u042462) |")

EX_2_OLD = "- **E39+**: free. The nearest candidate is ECA / epiplexity (\u041317)."
EX_2_NEW = "- **E40+**: free. The nearest candidate is ECA / epiplexity (\u041317)."

EDITS = {
    EV: [("EV-1 header date", EV_1_OLD, EV_1_NEW),
         ("EV-2 \u042461 and \u042462", EV_2_OLD, EV_2_NEW),
         ("EV-3 \u042446 caveats", EV_3_OLD, EV_3_NEW),
         ("EV-4 \u042460 caveats", EV_4_OLD, EV_4_NEW)],
    EX: [("EX-1 E39 row", EX_1_OLD, EX_1_NEW),
         ("EX-2 free slot to E40+", EX_2_OLD, EX_2_NEW)],
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
