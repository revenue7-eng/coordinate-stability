# E40: initialisation sweep at a fixed data seed

Separates the encoder initialisation from the data sample, which every earlier
experiment in this line varied together.

Facts produced: Ф64. Hypotheses touched: Г26 (split), Г27 (new).
Registry: EVIDENCE.md, EXPERIMENTS.md. Nothing is restated here.

## The question

E39 left an observation that looked like a trend: across its five seeds, the
initialisation with the most linearly extractable information about the true
state also gave the worst result. But there the initialisation and the data
sample varied together, so the observation could not be read either way.

Here the data seed is fixed at 42 and only the initialisation varies. The
encoder is frozen at step 0 and never trains, so best_vp measures how good that
fixed initialisation is as a coordinate system, while R2_readout and eff_rank
measure what it carries.

## Design note

`run_subepoch` seeds one global stream from the data seed, and the model
parameters are drawn from it, so initialisation could not be varied
independently. `e40_lib.py` adds an optional `init_seed`: the stream is
reseeded immediately before the model is built and restored to the data seed
immediately after, so that the DataLoader shuffling order is identical across
initialisations and only the parameters differ.

Because of that restore, E40 runs are comparable among themselves but not
directly against E39's step 0, where the stream was never touched.

`e40_lib.py` is a copy of `e39_lib.py`, which is itself a copy of
`e32_lib.py`. Each experiment owns its library so that registered results stay
reproducible as recorded. Both patch scripts assert that every substitution
matches exactly once.

## Files

```
code/run_e40.py         the sweep; usage: python run_e40.py [n_inits]
code/e40_lib.py         E40's library copy, init_seed separable
code/patch_e40_lib.py   creates e40_lib.py from e39_lib.py
results/sweep.json      per-initialisation best_vp, R2_readout, eff_rank
checkpoints_sha256.txt  manifest of the checkpoints
```

Checkpoints are kept locally; the repository excludes `*.pt` by policy. The
representation metrics are computed with `analyze_representation.py` from E39,
imported rather than copied.

## Reproducing

```
python code/patch_e40_lib.py    # only if e40_lib.py is absent
python code/run_e40.py 10
```

Resume-safe: initialisations already present in `sweep.json` are skipped. About
two minutes per initialisation on an idle machine, plus one data collection and
one prescribed reference run.
