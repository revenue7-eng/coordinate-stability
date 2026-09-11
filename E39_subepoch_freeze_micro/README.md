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
