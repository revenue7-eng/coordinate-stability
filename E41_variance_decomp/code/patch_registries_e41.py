#!/usr/bin/env python3
"""Register E41 in EVIDENCE.md and EXPERIMENTS.md.

Cyrillic identifiers are built from escape sequences, never typed literally:
a previous session lost an anchor to a confusion between two look-alike Cyrillic letters in truncated output.
"""
G, F = "\u0413", "\u0424"
EV = "/mnt/d/coordinate-stability/EVIDENCE.md"
EX = "/mnt/d/coordinate-stability/EXPERIMENTS.md"

NEW_FACTS = f"""
## {F}65 — E41: the spread between frozen initialisations belongs to the encoder
- [verified: python E41_variance_decomp/code/analyze_e41.py over E41_variance_decomp/results/grid.json, 2026-09-14] Crossed grid, 8 encoder seeds x 5 head seeds, data seed 42 fixed, encoder frozen at step 0, one run per cell. Two-way crossed random effects, df 7 / 4 / 28, interaction confounded with the residual. Variance share of best_vp: encoder 0.993, head 0.002, residual 0.005; F_enc = 1021.2, F_head = 4.92.
- [verified: same run] Stable across metric and scale: final_vp 0.994, mean_last3 0.992; on the log scale 0.982, 0.985, 0.978. Within a fixed encoder the five heads spread 1.05x to 1.25x, while encoder levels run 0.002365 to 0.009158, a spread of 3.87x over seeds 1..8 (the raw E40 spread over the same eight is 4.72x).
- [verified: e40_lib.py:120-125 and 136-138] The confound is real in the code: E40 reseeded one stream from init_seed immediately before M(enc, AE(), PR(), SIGReg()), so a single argument drew encoder and head together, and with the encoder frozen before the first step only the head ever trained. It is negligible in magnitude.
- [verified: run_e41.py acceptance cell, bit_exact True, rel diff 0.000e+00] Cell (enc_seed=1, head_seed=None) reproduces E40 init 1 at best_vp = 0.0031448905217346915 exactly, so the seed split did not move the stream and encoder seeds 1..8 are bit-identical to E40 inits 1..8.
- VERDICT: the riskiest assumption carried into {G}27 is closed negatively. The spread is not a property of the (encoder, predictor) pair; it is a property of the encoder.

## {F}66 — the encoder level is recoverable from a single run
- [verified: analyze_e41.py, 2026-09-14] corr(E40 best_vp over inits 1..8, E41 encoder level averaged over five heads) = +0.9951 pearson, +0.9762 spearman.
- CONSEQUENCE: a crossed design is not needed for the campaigns that follow. One head per initialisation measures the encoder level to within the head noise, which halves the cost of the {G}27 sweep.

## {F}67 — eff_rank is not a confirmed lead for {G}27
- [verified: recomputation over E40_init_sweep/results/sweep.json, 2026-09-14] corr(best_vp, eff_rank) = -0.2871 over all ten initialisations, -0.6994 over the eight used in the E41 grid, -0.7005 without init 9, +0.3919 without init 7. Spearman over ten: -0.1030, exact two-sided permutation p = 0.785.
- [verified: analyze_e41.py] Against the head-averaged encoder level over the same eight: pearson -0.7111, spearman -0.4524, exact p = 0.2675. Averaging the head noise out is not what moves the number; the choice of points is.
- [verified: same files] init 9 (eff_rank 1.5380, best_vp 0.001221) and init 7 (eff_rank 1.5755, best_vp 0.008927) share the lowest ranks with opposite outcomes, which is why the correlation swings with the subset.
- [verified: same files] corr(best_vp, R2_readout) = +0.0596 over ten and +0.2307 over the eight; spearman over ten -0.0061, exact p = 1.000. The negative part of {F}64 is unaffected.
- VERDICT: eff_rank is not a lead. Seeds 1..8 were chosen for the E41 grid on no criterion related to eff_rank, so reading the stronger correlation on that subset as evidence would be a selection artifact.
"""

EV_EDITS = [
    (f"EV-1 {G}27 lead demoted",
     f"- The available lead is the effective rank of the representation, corr -0.287 at n=10, in the direction of higher rank being better. eff_rank also varies by initialisation in E39 (1.61 to 2.70 of a maximum 3) and is close to flat across the opening window ({F}63), so it is a property of the initialisation rather than of training.",
     f"- The effective rank of the representation is not a confirmed lead: the correlation moves from -0.70 to +0.39 depending on which single initialisation of the ten is left out, so its direction is set by the choice of points rather than by the data ({F}67). eff_rank does vary by initialisation in E39 (1.61 to 2.70 of a maximum 3) and is close to flat across the opening window ({F}63), so it is a property of the initialisation rather than of training."),

    (f"EV-2 {G}27 falsifier escape route closed",
     "- Falsifier: a sweep with enough initialisations to settle whether eff_rank, or any other cheap property of the untrained encoder, predicts best_vp. If none does, the spread has to be attributed to the interaction with the downstream module rather than to the encoder alone.",
     f"- Falsifier: a sweep with enough initialisations to settle whether eff_rank, or any other cheap property of the untrained encoder, predicts best_vp. If none does, the property that orders the bases is not a cheap one; attributing the spread to the interaction with the downstream module is no longer available as an explanation ({F}65)."),

    (f"EV-3 {G}27 test renumbered",
     "- Test: E41. Initialisations at two or three data seeds, n large enough for a rank correlation to mean something, candidate predictors computed before training.",
     f"- Test: E42. About 30 initialisations at a fixed data seed, one head per encoder (justified by {F}66), candidate predictors computed before training. n=30 resolves |rho| from 0.49 at alpha=0.05 and power 0.80. E41 established that the question is well posed ({F}65)."),

    (f"EV-4 {G}27 status",
     "- Status: OPEN, untested\n- E40",
     f"- Status: OPEN. The premise is established rather than assumed: the spread is attributable to the encoder with a variance share of 0.993 ({F}65). The eff_rank lead is withdrawn ({F}67). Tested by E42.\n- E40, E41"),
]

EX_EDITS = [
    ("EX-1 E41 taken, E42 planned",
     f"- **E41+**: free. The nearest candidate is ECA / epiplexity ({G}17).",
     f"- **E41**: variance decomposition, encoder init x head init at a fixed data seed, frozen at step 0 (COMPLETE 2026-09-14, {F}65, {F}66, {F}67).\n"
     f"- **E42**: initialisation sweep for {G}27, about 30 encoders, one head each, candidate predictors computed before training (PLANNED).\n"
     f"- **E43+**: free. The nearest candidate is ECA / epiplexity ({G}17)."),
]

def apply(path, edits, append=None, guard=None):
    src = open(path, encoding="utf-8").read()
    if guard:
        assert guard not in src, f"{path}: guard {guard!r} already present, patch already applied?"
    counts = [(name, src.count(old)) for name, old, _ in edits]
    bad = [(n, c) for n, c in counts if c != 1]
    assert not bad, f"{path}: anchors not unique, nothing written: {bad}"
    for name, old, new in edits:
        src = src.replace(old, new)
        print(f"matched {name}")
    if append:
        src = src.rstrip("\n") + "\n" + append
        print(f"ok appended {len(append.splitlines())} lines")
    open(path, "w", encoding="utf-8").write(src)
    print(f"WRITTEN to disk: {path}")

apply(EV, EV_EDITS, append=NEW_FACTS, guard=f"{F}65")
apply(EX, EX_EDITS, guard="**E42**")

# anchors back out by code point, never by eye
for label, needle in ((f"{F}65", f"## {F}65"), (f"{F}66", f"## {F}66"),
                      (f"{F}67", f"## {F}67")):
    s = open(EV, encoding="utf-8").read()
    assert s.count(needle) == 1, f"{label}: expected 1 heading, found {s.count(needle)}"
    print(f"ok heading {label} U+{ord(label[0]):04X}{label[1:]}")
