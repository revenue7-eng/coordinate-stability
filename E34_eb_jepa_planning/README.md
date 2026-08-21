# Experiment 34: EB-JEPA — Prescribed vs Free on LeCun's Benchmark

## What this tests
Can prescribed axes (agent x,y coordinates) match free encoder (ImpalaEncoder CNN) for planning in a world model with obstacles? Tested on LeCun et al.'s EB-JEPA benchmark (Two Rooms, MPPI planner).

## Key result
**Prescribed 0% SR, Free 55% SR.** Prescribed axes with only agent coordinates cannot plan in environments with obstacles (walls, door). Hypothesis that prescribed axes are sufficient for downstream planning is **refuted** for this setting.

## Setup
- **Environment:** Two Rooms (EB-JEPA, Meta FAIR, 2602.03604)
- **Task:** Goal-conditioned navigation with MPPI planning in latent space
- **Architecture:** EB-JEPA with VICReg + IDM + temporal similarity regularization
- **Prescribed encoder:** MLP (2→256→256→512), input = agent (x, y), 199K params
- **Free encoder:** ImpalaEncoder CNN, input = 65×65 RGB pixels, 1.43M params
- **Predictor:** RNNPredictor, 793K params (shared)
- **Planner:** MPPI, 200 samples, 20 iterations, plan_length=90, 200 steps allowed
- **Seed:** 1
- **Epochs:** 12
- **Planning eval:** 20 episodes per condition, epoch 11
- **Platform:** Prescribed training — Colab Pro T4 GPU (~50h). Free training — Windows CPU, Python 3.14 (~150h). Planning eval — Colab Pro T4 GPU (~1.5h)

## Results

### Planning (main metric)
| Condition | Success Rate | Mean Distance |
|-----------|-------------|---------------|
| free | **55%** (11/20) | 9.78 |
| prescribed | **0%** (0/20) | 41.54 |

### Training (epoch 11)
| Condition | Pred Loss | Probe Loss | Encoder Params |
|-----------|-----------|------------|----------------|
| free | 0.0237 | 0.072 | 1,426,096 |
| prescribed | 0.0512 | 0.006 | 199,168 |

Pred loss: free 2.2× better. Probe loss is **not comparable across conditions**: the probe (MLPXYHead on detached latent, MSE against agent loc) recovers the encoder's own input in the prescribed condition, but must extract coordinates from pixels in the free condition. The two numbers measure different things.

### Key observations
- Prescribed probe converges to near-zero (0.006), starting from 9.77 at epoch 0 — recovery is guaranteed by construction but still has to be learned through the MLP + LayerNorm
- Free probe stagnates for 4 epochs then slowly falls — latent space becomes interpretable late
- Prescribed sim_loss increases while pred_loss decreases — temporal similarity regularization is counterproductive for prescribed space
- Prescribed failure is not uniform: per-episode distances range 8.36 to 71.37, with three episodes finishing at 8.36 / 9.12 / 13.30 — comparable to the free mean of 9.78. SR=0% means the success threshold was never met, not that every trajectory ran into the wall
- Free has no per-episode arrays: `planning_eval_results.json` holds aggregates only (160 bytes vs 896 for prescribed). The 11/20 split is documented in the April report, not recoverable from the committed artifact
- Free SR (55%) is below their reported ~97% — likely due to medium task config (200 steps) vs their optimized setup

### Why prescribed fails at planning
Prescribed encoder sees only agent (x, y). Two Rooms has a wall with a door dividing the space. Prescribed latent contains no information about the wall, and MPPI routes trajectories through it. Predictor learns correct dynamics (pred loss 0.051). B1 confirms the free side of this: the free latent encodes wall position almost perfectly (R2 0.969) while door position is barely readable (R2 0.211). The contrast between conditions is about the obstacle, not about the door. How free reaches the opening while representing its position so weakly is not answered here.

## B1: what is in the free latent

Linear probe from the frozen free encoder latent (512d, `final_ln` output) to the
room geometry. Ridge with cross-validated alpha, held-out R2, 4000 train and 1000
test episodes. Thresholds 0.7 and 0.3 were fixed before the run.

| Target | free | prescribed_2 | prescribed input (2d) |
|---|---|---|---|
| wall_x | **0.9691** | 0.0388 | 0.0091 |
| door_y | **0.2109** | 0.0336 | 0.0147 |

Permutation floor for free, refitting on shuffled labels: +0.0011 +/- 0.0090 for
wall_x, -0.0038 +/- 0.0034 for door_y. Stable across frames, t=8 gives 0.9687 and
0.2138. Ridge alpha settled at 0.215 for both targets, so the signal is not being
squeezed out from under heavy regularization.

Wall position clears the 0.7 threshold. Door position sits below 0.3 but is well
above its floor, so it is weak rather than absent.

The prescribed_2 columns are the structural control. That encoder is an MLP over
two numbers, so its latent cannot carry the wall in any richer form than those two
numbers allow. It yields no more than its own input (gap 0.0297 for wall_x, below
the 0.10 flag), which shows the measurement itself does not manufacture the free
result.

Three limits on these numbers:

- A linear probe measures linear decodability, not presence of information. door_y
  at 0.211 means it does not read out linearly. It does not mean the door is absent
  from the latent.
- Episodes are freshly drawn from the E34 configuration, not the ones E34 trained
  on. `init_data` runs before `setup_seed` in `run_experiment_v3_windows.py:370-373`,
  so that draw was never seeded and cannot be reproduced.
- The control checkpoint is epoch 0. For a claim about what the input can carry
  this makes the check stricter, not weaker.

## Files
```
code/eb_jepa_v3.ipynb                          — Training notebook (Colab)
code/run_experiment_v3_windows.py              — Training script (Windows)
code/eb_jepa_planning_eval.ipynb               — Planning eval notebook (Colab)
results/free/training_results.json             — Free training metrics, 12 epochs
results/free/planning_eval_results.json        — Free planning: SR=55%, 20 episodes
results/free/encoder_stats.json                — Free encoder statistics per epoch (from working tree, not in the E32 archive)
results/prescribed/training_results.json       — Prescribed training metrics, 12 epochs
results/prescribed/planning_eval_results.json  — Prescribed planning: SR=0%, 20 episodes
results/prescribed/encoder_stats.json          — Prescribed encoder statistics per epoch
code/b1_latent_physics.py                      — B1 probe, free latent
code/b1_control_prescribed.py                  — B1 control, prescribed latent
results/b1_output.txt                          — B1 output, n=4000/1000
results/b1_control_output.txt                  — B1 control output, n=4000/1000
```

## Checkpoints (external, not in git)
Both conditions produced `latest.pth.tar` checkpoints, kept outside the repository:
- free: 33.8 MB, epoch 11, used for B1
- prescribed: 16.9 MB, epoch 0 step 1561

The available prescribed checkpoint holds epoch 0, not the epoch 11 that produced SR = 0%. The Н1 numbers come from `planning_eval_results.json`, not from weights, so they are unaffected. Whether epoch 11 prescribed weights still exist anywhere is open.

These are required for the planning eval and for any probing analysis on the latents.

## How to reproduce
```bash
# 1. Extract EB-JEPA code
unzip eb_jepa_colab.zip -d eb_jepa_free && cd eb_jepa_free
pip install -e .

# 2. Train free baseline
python experiments/prescribed_axes/run_experiment_v3_windows.py --mode free

# 3. Train prescribed
python experiments/prescribed_axes/run_experiment_v3_windows.py --mode prescribed

# 4. Planning eval (requires GPU, use Colab notebook)
# Upload checkpoints to Drive, run eb_jepa_planning_eval.ipynb
```
