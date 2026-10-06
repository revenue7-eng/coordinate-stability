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

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

**Ф64. What a fixed initialisation carries does not predict how good a coordinate system it is (E40)**
- STATUS: corrected by Ф75. The null is a property of the self-referential metric (the target is the encoder's own latent); on an external target R2_readout orders the frozen encoders (r = -0.86, n = 30, one data seed). On the own-latent metric the frozen initialisations are ordered by persistence (Ф73).
- Design: the data seed is held at 42 and only the encoder initialisation varies (10 initialisations). The encoder is frozen at step 0, so it never trains. The DataLoader stream is restored after construction, so batch order is identical across initialisations and only the parameters differ. This is the first measurement in the line that separates initialisation from data sample.
- best_vp by initialisation 1 to 10: 0.00314, 0.00399, 0.00386, 0.00483, 0.00189, 0.00273, 0.00893, 0.00354, 0.00122, 0.00352. R2_readout over the same: 0.2385, 0.3377, 0.4094, 0.4716, 0.2494, 0.5156, 0.4115, 0.5137, 0.4972, 0.3886.
- **corr(best_vp, R2_readout) = +0.060** over a two-fold range of R2_readout (0.239 to 0.516). Informativeness of a frozen initialisation does not predict the downstream result. The inverse ordering visible in the five E39 seeds was an appearance produced by initialisation and data sample varying together.
- best_vp nevertheless spreads **7.31x** across initialisations (0.00122 to 0.00893) at a fixed data sample. Fixed bases differ strongly from each other; what separates them is not what they carry.
- The prescribed encoder scores 0.00261 on the same data, inside the range of the random ones, with 2 of 10 initialisations better than it. The prescribed coordinate readout is not distinguished among frozen random bases (strengthens Ф31, Ф60).
- corr(best_vp, eff_rank) = -0.287, in the direction of higher rank being better, but not significant at n=10. Recorded as a lead, not a claim.
- Environment: Push-T real gym-pusht, data seed 42, EP=15/NEP=200, freeze at step 0, e40_lib (init_seed separable), local CPU.
- Caveats: one data sample only, so the 7.31x spread is within-sample across initialisations and its dependence on the sample is unmeasured. n=10 supports the null on R2_readout but not a claim about eff_rank.
- Artefact: E40_init_sweep/ (README + code + results/sweep.json)
- E40

---
---

## Facts
Ф64

## Status
Summary table: null corrected by Ф75.

- Ф64: STATUS: corrected by Ф75. The null is a property of the self-referential metric (the target is the encoder's own latent); on an external target R2_readout orders the frozen encoders (r = -0.86, n = 30, one data seed). On the own-latent metric the frozen initialisations are ordered by persistence (Ф73).
