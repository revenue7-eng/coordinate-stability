# Re-evaluation 2026-08-24

Planning eval of the free (pixel) encoder, epoch 11, 20 episodes, CPU.
Produced by `E35_eb_jepa_prescribed4/code/planning_eval_v4.py`
(md5 of the executed file: `3730fe87ad8473ed6289796f1812ba38`; the committed
copy differs by the removal of one unused import, `main_eval`).

The companion run for prescribed_2 is in `../prescribed_reeval_20260824/`.
Both branches were measured by the same accounting: the episode always runs
the full `n_allowed_steps`, and success is the state of the final step, not
the fact of having reached the goal at some point.

## Files

- `planning_eval_results.json` — per-episode successes, distances and episode
  geometry, plus aggregates. md5 `89368f1de60f0e196087033e169097ec`.
- `free_ep20.log` — full run log, 20 episodes. md5 `cdb853002170eb25db6913508b7fc621`.
- `free_ep1.log` — earlier single-episode run. md5 `dfa510f8a9e43572c4d421df550cbbd7`.

Logs are stored byte-for-byte as downloaded from the pod, so the checksums
above stay verifiable. They contain tqdm progress output written with carriage
returns, which makes them unreadable in a plain viewer. To read one:

    tr '\r' '\n' < free_ep20.log | grep -E '^\[INFO|ep [0-9]+:'
