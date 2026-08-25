# Re-evaluation 2026-08-24

Planning eval of the prescribed_2 encoder, epoch 11, 20 episodes, CPU.
Produced by `E35_eb_jepa_prescribed4/code/planning_eval_v4.py`
(md5 of the executed file: `3730fe87ad8473ed6289796f1812ba38`; the committed
copy differs by the removal of one unused import, `main_eval`).

The companion run for the free encoder is in `../free_reeval_20260824/`.
Both branches were measured by the same accounting: the episode always runs
the full `n_allowed_steps`, and success is the state of the final step, not
the fact of having reached the goal at some point. The two runs saw the same
twenty episode geometries.

## Files

- `planning_eval_results.json` — per-episode successes, distances and episode
  geometry, plus aggregates. md5 `109156a7867a0012c7c6fc3ebcf3cc33`.
- `ep20.log` — full run log, 20 episodes. md5 `8265d30ede6b2a63edb43856d5a5b014`.
- `ep1.log` — single-episode run. md5 `fccee5b48beb3acc1cae350e784059d3`.

Logs are stored byte-for-byte as downloaded from the pod, so the checksums
above stay verifiable. They contain tqdm progress output written with carriage
returns, which makes them unreadable in a plain viewer. To read one:

    tr '\r' '\n' < ep20.log | grep -E '^\[INFO|ep [0-9]+:'
