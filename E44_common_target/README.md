# E44: Common-target comparison at the E28 dim-5 point (synthetic Push-T)

## What this tests
Whether the E28 dim-5 prescribed-against-free gap survives scoring on a common external target, with four arms: prescribed, prescribed_rotated, free_raw and free_scaled.

## Key results
From EVIDENCE.md, verbatim; the registry is authoritative.

## Ф76: the free encoder on the raw Push-T state discards the length of the positional vector and nearly all of the block angle
- [verified: python3 E44_common_target/code/fe_input_scale_probe.py; refutation criterion fixed in the file before the first run: median raw ratio >= 0.5 on either test] 40 frozen E42 encoders (30 at data seed 42, 10 at 123), 1220 gym-pusht states. Output change when (x_a, y_a, x_b, y_b) is scaled by 0.95, relative to an orthogonal perturbation of the same norm: median 0.0060 (seed 42, range 0.0023 to 0.0120) and 0.0067 (seed 123). Output change when the block is rotated by 10% of its range, relative to moving it by 10% of its range: median 0.0153 (seed 42, range 0.0057 to 0.0375) and 0.0200 (seed 123). First-layer bias share ||b|| / ||Wx||: 0.0017 and 0.0018. The same weights on input divided by (512, 512, 512, 512, 2 pi): 0.65, 0.92, 0.76 (seed 42) and 0.86, 1.14, 0.77 (seed 123).
- Mechanism: FE is Linear(5,64), LayerNorm, GELU, Linear(64,64), LayerNorm, GELU, Linear(64,3). On coordinates in [0, 512] the first-layer bias is negligible and LayerNorm removes the scale of Wx, so the output depends on the direction of the positional vector only. The angle, in [0, 2 pi], moves the output between about 25 and 175 times less than an equal fraction of a coordinate range. The defect is in the input, not in the weights. [verified: e43_lib.py:73-79; the raw state reaches the encoder through DS and run_subepoch, e43_lib.py:56-66 and 109-160]
- Scope: the same FE definition is in E06, E07, E31, E32 and E39 to E43 [verified: grep -rl "class FE" --include=*.py], and the E28 FreeEncoder is the same network on the raw state [verified: p2_dim_sweep_full.py:81-94, 151-166]. The raw input path is verified for E39 to E43 [verified: DS appends st[t:t+H+2] unscaled and M.forward calls s.enc(st), grep -F in e39_lib.py and e40_lib.py; e43_lib.py is a byte-identical extension of e42_lib.py and e41_lib.py] and for E28. For E06, E07, E31 and E32 only the definition is verified. In every case PE divides by the range.
- CONSEQUENCE: the frozen-encoder line (Ф64 to Ф75, paper 1) measures spread inside this family. Nothing in it has been shown for encoders with scaled input. R2_readout on PE is in effect a readout of block x and y through the direction of the positional vector, since the angle reaches no FE with more than a few percent of the weight of a coordinate shift.

## Ф77: E28 scores prescribed and free on each encoder's own latent, with free on the raw state
- [verified: p2_dim_sweep_full.py, WorldModel.forward and val_loss] The loss is mse(prediction, emb[:, H]), where emb is the output of the encoder under test, and val_loss averages it. The prescribed latent is make_prescribed_features (no parameters; coordinates divided by 512, angle by 2 pi). The free latent is learned under SIGReg. The two losses are in different units.
- [verified: Ф76] FreeEncoder receives the raw state; PrescribedEncoder receives it range-normalised.
- Precedent: the 12x probe figure for Н1 was withdrawn for the same kind of incomparability (Н1, run_experiment_v3_windows.py:490-494).
- Observation, cause not established: the E28 ratio is largest where the prescribed latent holds only the block (dim 2: 1820x, dim 3: 228x) and smaller where it includes the agent (dim 5: 66x, dim 11: 42x). A self-referential loss ranks by latent step size (Ф73), and the block moves little between steps: on gym-pusht the persistence baseline explains 96% of the variance of the block pose at t+3 (E44_common_target/code/feasibility_oracle.py). [verified: python3 E44_common_target/code/e28_persistence_check.py] On E28's own synthetic dynamics the block moves on 4.4% to 4.7% of recorded steps (seeds 42, 123, 777). The one-step persistence MSE of the prescribed latent is about 0.000001 at dim 2 and about 0.004 at dim 5. At dim 2 the E28 prescribed loss (0.000004 to 0.000006) is above that persistence on all three seeds while the E28 ratio to free is 1539x to 2175x; persistence is over all windows, the E28 loss is the best epoch on a 10% validation split. Ordered by mean persistence and by mean ratio, the seven dims agree except for dims 1, 5 and 7 (Spearman -0.89, descriptive). The pattern is what the units alone would produce.
- CORRECTS Ф17, Ф18, Ф36: the ratios 42x to 1820x are not interpretable as a difference in quality between the encoders until both are scored on a common target with equally scaled input. They are not refuted: the direction on a common target is unknown. The same holds for the "Gap prescribed/free" row of KEY DIFFERENCES BETWEEN ENVIRONMENTS and for П1 and П2 as far as they rest on these ratios.
- Likely affected, not checked: Ф37 (gauge fixing, the same kind of loss comparison), Tier 3 E25 (Ф18 reports agreement with E28 at dim 5, 66.3x against 66.2x), and the double-pendulum ratios of Ф20.
- Not affected: Ф58 (E35, planning SR is a metric common to both encoders).
- NEXT: E44, pre-registered in EXPERIMENTS.md: four arms at the E28 dim-5 point on a common external target.
- At dim 5, Ф78 decomposes the E28 ratio: about 37x is the input defect (free_raw / free_scaled on the common target) and about 1.4x the units (60x own-latent best against about 42x common-target final; the two use different epoch selections, so the split is approximate).

## Ф78: E44, on a common target the E28 dim-5 gap reduces to the input defect; registered class O4 PROVISIONAL
- STATUS (Ф83): the prescribed arm here is range-normalised, not standardised; its scale handicaps the predictor. Does not test fixation as such.
- [verified: python3 E44_common_target/code/analyze_e44.py over E44_common_target/results/cells/seed_*.json; pre-registration 483f39e] 10 seeds, four arms. Gates all true: one pre-registration commit, ancestor of HEAD; prescribed stage 2 equal to stage 1 on every seed; encoder state unchanged by stage 2 in every cell.
- Registered statistic: GMR of final2(free_scaled) / final2(prescribed) 1.146, 95% CI [0.937, 1.402], t-test p 0.159. Class O4. Median convergence 0.863 (prescribed) and 0.873 (free_scaled), below 0.95: the class is PROVISIONAL and is not read as a difference in quality. Both arms were still improving at epoch 30, at similar rates.
- [verified: same] Named contrasts: free_raw / free_scaled 36.6 [29.5, 45.4]; prescribed_rotated / prescribed 1.002 [0.853, 1.178]; E28-style free_raw / prescribed on stage-1 best 60.2 [48.6, 74.4] (E28: 66x); registered statistic on best 1.238 [1.122, 1.367], class O2.
- Run conditions: workers ran with OMP_NUM_THREADS=1 (threads 1 in every seed file, torch 2.13.0+cpu); acceptance ran with the default thread count. free_raw stage-1 best at seed 42 is 0.023911051360661524 in the campaign against 0.023911124657382044 at acceptance (relative 3.1e-6). A first launch with default threads in four parallel processes oversubscribed the four cores and was stopped before any epoch finished. A WSL crash interrupted the second launch about a minute in, before any seed file was written; the campaign was relaunched from scratch with the same commit and seeds.
- INTERPRETATION: on the common target free_raw is about 42x worse than prescribed (36.6 x 1.146), close to its 60x on own latents, so at dim 5 the E28 gap is mostly the input defect of Ф76 rather than the units of Ф77. With equally scaled input the remaining prescribed advantage lies between none and about 1.4x on final (1.1x to 1.4x on best), and it does not come from axis alignment with the target.
- CORRECTS Ф17, Ф18, Ф36 at dim 5: an order-of-magnitude advantage of prescribed over free is not supported once free receives a range-normalised input; the 66x of Ф18 at dim 5 measures mostly the input defect. The other dims remain not interpretable.
- NEXT: E45, pre-registered in EXPERIMENTS.md: prescribed and free_scaled at 90 epochs per stage.

## Setup
From EXPERIMENTS.md, section E44.

Question (Ф77). E28 reports prescribed beating free by 66x at dim 5, on each encoder's own latent and with free on the raw state (Ф76). Is that a difference in quality, or a difference in units and input scaling?

Preconditions, met before this block was committed. At acceptance, e44_lib.py was p2_dim_sweep_full.py byte for byte plus an appended block (results/acceptance_lib.sha256 holds its sha256 and size). The committed e44_lib.py is that file plus a second appended block (the rotated arm). Acceptance (results/acceptance.log): stage 1 reproduced E28 prescribed_dim5_seed42 bit for bit (0.00032244045797964196). free_dim5_seed42 was reproduced to a relative difference of 8.7e-07 (0.023911124657382044 against E28 0.023911103972217494), not bit for bit. A second run of free_raw stage 1 with the committed library (results/determinism_free_raw.log) reproduced 0.023911124657382044 bit for bit, so the current environment is deterministic and the difference from E28 is attributed to the environment E28 ran in, which is not identified [INFERENCE]. Prescribed stage 2 equalled its stage 1 bit for bit. The tolerance for free_raw (relative 1e-5) was set after seeing the acceptance line and is recorded as such; it is five orders of magnitude below the resolution the registered statistic needs. No common-target loss of a free or rotated arm was computed before this commit.

Setup. dim 5, EPISODES 200, EPOCHS 30, data synth(200, seed); split, loaders, optimiser, SIGReg weight 0.09 and loop as in E28.
Arms:
- prescribed: PrescribedEncoder (state divided by its range), as in E28.
- prescribed_rotated: the same features centred, multiplied by a fixed random orthogonal 5x5 matrix (private generator, seed 20260928) and shifted back.
- free_raw: FreeEncoder on the raw state, as in E28.
- free_scaled: the same network on the state divided by (512, 512, 512, 512, 2 pi); same random stream as free_raw.
Stage 1: E28 run_condition at dim 5, own-latent target. Stage 2: encoder frozen, fresh action encoder and predictor, same seed, same loop, target make_prescribed_features(s_{t+3}, 5), a fixed function of the true state.
Seeds: 42, 123, 777 (E28) and 1001 to 1007, n = 10. All arms of a seed share data and split.

Known asymmetry. The common target equals the prescribed latent: the prescribed predictor sees its target coordinates at t..t+2 and has only the dynamics to learn, while a free predictor must also decode them. The comparison is an upper bound in favour of prescribed: O4 or O5 would be strong, O1 weaker than it sounds. prescribed_rotated measures how much of a prescribed advantage is axis alignment with the target.

Registered statistic. Per seed L = ln(final2(free_scaled) / final2(prescribed)), where final2 is the stage-2 validation loss at the last epoch (no selection on validation). Mean L with a 95% t interval (df 9), mapped through exp to the geometric-mean ratio GMR and its interval [lo, hi]. The class alone decides; the t-test p is reported only.

Outcomes:
- O1 order of magnitude survives: lo >= 10.
- O2 prescribed better, not by an order of magnitude: 1 < lo and hi < 10.
- O3 prescribed better, magnitude unresolved: 1 < lo < 10 <= hi.
- O4 no difference resolved: lo <= 1 <= hi.
- O5 free better: hi < 1.
The threshold 10 is the programme's claim "order of magnitude" (Ф36). The E28 figure 66x is a best-epoch loss on each encoder's own latent and is not directly comparable to GMR.

Convergence. For each arm conv = mean(stage-2 loss over epochs 26 to 30) / mean(epochs 21 to 25). If the median over seeds of conv for prescribed or free_scaled is below 0.95, the class is reported as PROVISIONAL (not converged) and is not read as a difference in quality.

Validity gates, the campaign is void if any fails: on every seed prescribed stage 2 equals stage 1 bit for bit; the encoder state is tensor-equal before and after stage 2 in every cell; every seed file carries this pre-registration commit and it is an ancestor of HEAD.

Named, not deciding (each as GMR with its 95% interval over seeds):
- free_raw / free_scaled on final2: the contribution of the input defect (Ф76).
- prescribed_rotated / prescribed on final2: the contribution of axis alignment.
- free_raw / prescribed on stage-1 best: the E28-style unit-confounded ratio, expected near E28.
- the registered statistic on best instead of final, with its class.

Scope: dim 5 and E28's synthetic dynamics only. Ф17 (dim 11) and the other Ф18 dims stay "not interpretable".

Code: E44_common_target/code/e44_lib.py, run_e44.py (runs only if this commit is an ancestor of HEAD, holds this block, touches the three code files, and the code is unchanged since), analyze_e44.py. Results: E44_common_target/results/cells/seed_<s>.json, analysis.json.

Cost, from acceptance.log: prescribed cell (both stages) 456 s, free_raw stage 1 461 s. Per seed about 2 x 456 + 4 x 461 = 2756 s (46 min). Ten seeds on four workers (3, 3, 2, 2 seeds): about 2.3 h wall if the processes do not slow each other.

Result (Ф78): O4 PROVISIONAL. GMR 1.146 [0.937, 1.402]; median convergence 0.863 and 0.873, below 0.95. Workers ran with one thread each (see Ф78).

## Files
- `code/__pycache__`
- `code/acceptance_e44.py`
- `code/acceptance_progress.py`
- `code/analyze_e44.py`
- `code/determinism_free_raw.py`
- `code/e28_persistence_check.py`
- `code/e44_lib.py`
- `code/fe_input_scale_probe.py`
- `code/feasibility_oracle.py`
- `code/patch_prereg_e44.py`
- `code/run_e44.py`
- `results/acceptance.log`
- `results/acceptance_lib.sha256`
- `results/analysis.json`
- `results/cells`
- `results/determinism_free_raw.log`
- `results/worker_1.exit`
- `results/worker_1.log`
- `results/worker_2.exit`
- `results/worker_2.log`
- `results/worker_3.exit`
- `results/worker_3.log`
- `results/worker_4.exit`
- `results/worker_4.log`

## Facts
Ф76, Ф77, Ф78

## How to reproduce
```
OMP_NUM_THREADS=1 python3 -u E44_common_target/code/run_e44.py --prereg 483f39e5ba527e84188a8975d6e3017249de18d4 --seeds 42 123 777 1001 1002 1003 1004 1005 1006 1007
python3 E44_common_target/code/analyze_e44.py
```

## Status
Summary table: O4 PROVISIONAL.

- Ф76: no correction recorded in the registry; stands as recorded
- Ф77: no correction recorded in the registry; stands as recorded
- Ф78: STATUS (Ф83): the prescribed arm here is range-normalised, not standardised; its scale handicaps the predictor. Does not test fixation as such.

<!-- marker: repo-state v3 applied -->
