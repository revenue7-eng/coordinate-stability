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
Prescribed encoder sees only agent (x, y). Two Rooms has a wall with a door dividing the space. Prescribed latent space contains no obstacle information. MPPI planner cannot find paths through the door because the door does not exist in prescribed space. Predictor learns correct dynamics (pred loss 0.051) but planner routes trajectories through walls.

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
```

## Checkpoints (external, not in git)
Both conditions produced `latest.pth.tar` checkpoints, kept outside the repository:
- free: 33.8 MB
- prescribed: 16.9 MB

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
