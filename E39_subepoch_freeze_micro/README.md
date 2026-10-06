# E39: sub-epoch freeze micro-grid

Resolves the opening window of epoch 1 at single-optimizer-step resolution, to
decide whether the damage measured by E30/E31/E32/E38 is coordinate-basis drift
or the collapse-then-recover transient that T-JEPA/I-JEPA report in the first
iterations of training.

Facts produced: Ф61 (confound closed), Ф62 (seeding defect in e32_lib).
Registry: EVIDENCE.md, EXPERIMENTS.md. Nothing is restated here.

## Why E38 could not answer this

E38's finest grid point is f=0.05, which at n_batches=160 is optimizer step 8.
The entire window under test lies below that resolution.

## Design

Grid is specified in optimizer steps, not fractions: {0, 1, 2, 3, 4, 6, 8}.
Fractions are derived as f=(step+0.5)/n_batches, so that the library's
floor(f*n_batches) lands on the intended step regardless of float
representation. Steps 0 and 8 use E38's literal f values (0.00, 0.05).

5 seeds {42, 123, 777, 2024, 7}, EP=15/NEP=200, real gym-pusht, local CPU.

The discriminating prediction is about shape, not magnitude. Collapse followed
by recovery requires a recovery segment: a rise to a peak, then a sustained
fall. The encoder is frozen at step f and stays frozen for all remaining
epochs, so it cannot recover from a dip, and such a segment would be visible.

## Files

```
code/run_seed.py            per-seed runner; usage: python run_seed.py <seed>
code/e39_lib.py             E39's copy of the E32 training library (see below)
code/patch_e39_lib.py       creates e39_lib.py from e32_lib.py, adds checkpointing
code/patch_e39_seedfix.py   adds the action_space seeding fix (Ф62)
code/run_<seed>.log         console output of each run
results/seed_<seed>.json    full run_subepoch return per grid point
checkpoints_sha256.txt      manifest of the checkpoints
prefix_bug/                 seed 42 measured before the seeding fix, retained
```

`e39_lib.py` is a copy, not an edit. `e32_lib.py` is shared with E32 and E38,
whose numbers are already in the registries, so it is left untouched and its
defect is recorded rather than fixed in place. Both patch scripts assert that
each substitution matches exactly once.

Checkpoints (70 files, 2 per grid point: the encoder at the moment of freezing
and the full model after all epochs) are kept locally. The repository excludes
`*.pt` by policy; `checkpoints_sha256.txt` is committed instead. Since the
encoder stays frozen from step f onward, the encoder inside `_model_final.pt`
is the representation at step f, and `_enc_at_freeze.pt` records it directly so
that this does not have to be inferred.

## Reproducing

```
python code/patch_e39_lib.py       # only if e39_lib.py is absent
python code/patch_e39_seedfix.py   # only if the fix is absent
python code/run_seed.py 42
```

Runs are resume-safe: grid points already present in the seed JSON are skipped.
Expect `n_batches observed: {160}` and no `MISMATCH` lines. One seed takes
roughly 13 minutes on an idle machine; CPU contention (for instance a bitbake
build in a parallel WSL distribution, which shares the core pool) stretches
that to well over an hour without affecting the numbers.

Data collection is reproducible only through `e39_lib`. The same call through
`e32_lib` returns a different dataset on every invocation, including twice
within one process.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

**Ф61. The JEPA initial-collapse confound does not explain the sub-epoch damage (E39)**
- STATUS: see Ф87. Own-latent target.
- Question: inside epoch 1, is the measured quantity coordinate-basis drift, or the collapse-then-recover transient that T-JEPA/I-JEPA report in the first iterations? E38's finest point (f=0.05) is optimizer step 8 of 160, so the whole opening window sat below its resolution.
- Design: grid specified in optimizer steps {0,1,2,3,4,6,8} rather than fractions, f=(step+0.5)/n_batches so that floor(f*n_batches) lands on the intended step regardless of float representation. n_batches=160 (measured), EP=15/NEP=200, 5 seeds {42,123,777,2024,7}, real gym-pusht, local CPU.
- Discriminating prediction: collapse-then-recover requires a recovery segment, a rise to a peak followed by a sustained fall. The encoder is frozen at step f and stays frozen for all remaining epochs, so it cannot recover from a dip, and such a segment would be visible. **No seed shows one.**
- best_vp over steps 0,1,2,3,4,6,8: seed 42 0.00330/0.00333/0.00339/0.00347/0.00357/0.00388/0.00430; seed 123 0.00181/0.00180/0.00184/0.00185/0.00188/0.00200/0.00213; seed 777 0.00282/0.00312/0.00340/0.00374/0.00401/0.00486/0.00587; seed 2024 0.00101/0.00103/0.00110/0.00105/0.00110/0.00113/0.00126; seed 7 0.00843/0.00846/0.00853/0.00858/0.00859/0.00885/0.00923.
- Strictly monotone in **3/5** seeds (42, 777, 7). Two seeds show a single dip at different steps (123 at step 1, -0.6%; 2024 at step 3, -4.5%), each followed by continued rise. Do NOT state 5/5 for this window.
- The dips are not measurement noise: run_subepoch is deterministic given (eps, seed), verified by two identical runs in one process (0.003017362545391447 twice). Each point is an exact value, so a dip is a property of the pair (sample, step).
- Shape is invariant to the sampling defect of Ф62: the same grid on two independent samples for seed 42 (before and after the fix) gives window ratio 1.23x and 1.30x, monotone in both.
- Supported by Ф45: there is no recovery after epoch 1 either (freeze@1 to unfrozen = 1.3x).
- Consequence: the open confound recorded in Ф60's caveats is closed. The E30/E31/E32/E38 line measures damage that accumulates and persists, not a transient that resolves.
- Caveats: the window ratio varies strongly by sample (1.10x to 2.08x), and the between-sample spread at step 0 is 8.3x (0.00101 to 0.00843). Nothing below one optimizer step is resolved. The absence of a recovery segment is established for this architecture (predictor with stop-grad plus SIGReg, no EMA target encoder), not for JEPA variants in general.
- Artefact: E39_subepoch_freeze_micro/ (README + code + per-seed results). Checkpoints (70 files, 2 per grid point: encoder at freeze and full model after all epochs) are kept locally and not in the repository, which excludes *.pt by policy; their SHA256 manifest is committed as E39_subepoch_freeze_micro/checkpoints_sha256.txt.
- E39

**Ф62. collect_gym_data in e32_lib does not reproduce for a given seed (action space unseeded)**
- The function seeds its own Generator and uses it for env.reset, the branch draw and the noise draw, but `env.action_space.sample()` draws from the action space's own generator, which gym.make initialises from system entropy. That branch fires on the first step of every episode and in roughly 30% of later steps, so about a third of all recorded actions came from an unseeded source.
- Probe: two calls with seed=42 in one process give different SHA256 of the pickled episodes, and a second process gives two more distinct hashes (4 of 4 different). After adding `env.action_space.seed(int(rng.integers(0, 100000)))` all four agree (859c6a33fc96a239).
- Magnitude: at nominal seed 42 and f=0.00, three processes produced best_vp 0.00302, 0.00327 (E38 as shipped) and 0.00332, a spread of about 10%.
- Scope, affected: absolute numbers from E32 and E38, and any cross-run comparison of them. The label "5 seeds" in Ф46 and Ф60 means five independent samples whose labels do not identify them, not five controlled repetitions of one condition; the reported spread is between-sample.
- Scope, NOT affected: every shape verdict and within-run comparison. A run collects its data once and all grid points of that run share it, so the curve shape is measured on one fixed sample.
- Decision: E38 is not rerun. The load-bearing claims are shape claims and are unaffected, the absolute anchors already carry a do-not-compare caveat, and E39 demonstrated shape invariance across the defect directly. The fix lives in E39_subepoch_freeze_micro/code/e39_lib.py; e32_lib.py is left untouched so that E32/E38 artefacts remain reproducible as recorded.
- Also refutes the inherited reasoning that thread count cannot affect a result "because the seed is deterministic". The premise was false; measured, threads change the ninth decimal only (0.003017362545391447 at 4 threads vs 0.003017361605899376 at 1), so the conclusion happened to hold.
- Artefact: E39_subepoch_freeze_micro/code/patch_e39_seedfix.py; E39_subepoch_freeze_micro/prefix_bug/ (results of the defective collection, retained as evidence)
- E39

**Ф63. Inside the opening window the representation gains linear information about the true state while the downstream result gets worse (E39)**
- Probe: for each grid point, the encoder checkpoint taken at the moment of freezing is run over that run's own validation states, and three quantities are computed. R2_readout is the R2 of a least-squares linear map from the 3-dimensional representation to the true scaled coordinates. Procrustes is the disparity against the previous grid point after centering, scaling and optimal rotation. eff_rank is the exponentiated entropy of the representation covariance spectrum (maximum 3).
- R2_readout rises monotonically over steps 0 to 8 in **5/5 seeds**, including the two whose best_vp is not monotone: 0.336 to 0.381 (seed 42), 0.424 to 0.452 (123), 0.337 to 0.367 (777), 0.311 to 0.355 (2024), 0.444 to 0.472 (7).
- Over the same steps best_vp rises as well (Ф61). Information and error increase together, so what improves in the representation is not what determines the result.
- At step 0 an untrained random encoder already reaches R2_readout 0.31 to 0.44 and performs within one order of magnitude of the prescribed encoder, which is a parameter-free readout of the true state (PE is x[...,2:5]*sc) and therefore scores 1 by construction (Ф31, Ф60).
- Procrustes disparity from the previous point is small over steps 1 to 4 (0.0004 to 0.0021) and 2 to 4 times larger at steps 6 and 8 in 5/5 seeds: the basis accelerates.
- eff_rank is set by initialisation (1.61, 1.79, 2.09, 2.09, 2.70) and is close to flat across steps. The only visible decline is seed 2024, 2.70 to 2.54, which is also the only seed with a pronounced dip in best_vp. One case, no claim.
- The prediction recorded before the computation was that R2_readout would hold constant, on the reading that only the basis moves while content is preserved. It is refuted: the content changes too, and monotonically.
- Sanity: the prescribed encoder scores exactly 1.0000 in the same pipeline.
- Environment: no training, forward passes only; validation split reproduced with the run's own DS and random_split generator seed; e39_lib, local CPU.
- Artefact: E39_subepoch_freeze_micro/code/analyze_representation.py, E39_subepoch_freeze_micro/analysis/repr_seed_*.json
- E39

## Facts
Ф61, Ф62, Ф63

## Status
Summary table: Ф62 seeding bug stands.

- Ф61: STATUS: see Ф87. Own-latent target.
- Ф62: no correction recorded in the registry; stands as recorded
- Ф63: no correction recorded in the registry; stands as recorded
