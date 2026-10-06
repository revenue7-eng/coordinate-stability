# Experiment 29: Noise Control (prescribed plus matched noise against free)

## What this tests
Whether the drift of the free encoder behaves like generic noise. Noise is added to the prescribed latent during training at amplitudes matched to the drift measured in E06, either independently per sample (i.i.d.) or as one displacement per epoch (correlated), and the downstream predictor is scored on the clean prescribed latent.

## Key results
| Condition | Seed 42 | Seed 123 | Seed 777 | Mean | vs prescribed |
|---|---|---|---|---|---|
| prescribed | 0.000049 | 0.000043 | 0.000020 | 0.000037 | 1.0x |
| noise_late | 0.000057 | 0.000056 | 0.000018 | 0.000044 | 1.2x |
| noise_mid | 0.000264 | 0.000316 | 0.000119 | 0.000233 | 6.2x |
| noise_early | 0.031816 | 0.031516 | 0.032010 | 0.031781 | 851x |
| noise_schedule | 0.000070 | 0.000078 | 0.000029 | 0.000059 | 1.6x |
| correlated_mid | 0.000060 | 0.000069 | 0.000013 | 0.000047 | 1.3x |
| correlated_schedule | 0.000056 | 0.000074 | 0.000019 | 0.000050 | 1.3x |
| free | 0.008136 | 0.008988 | 0.007722 | 0.008282 | 222x |

- i.i.d. noise at the amplitude of early drift degrades the downstream predictor by 851x (Ф41i).
- A constant shift per epoch barely harms it, 1.3x even at the largest amplitude: the predictor compensates for a displacement that preserves differences between timesteps (Ф42i).
- The noise arms are all scored in the same clean prescribed latent, so they compare with each other. The free row is scored on its own latent and does not compare with them (Ф82).

## Setup
- **Environment:** synthetic Push-T, 200 episodes
- **Seeds:** 42, 123, 777
- **Epochs:** 30, prediction loss only (no SIGReg), architecture as in the tier 1 and tier 2 tests
- **Noise matching:** drift from E06 is the mean l2 displacement of validation embeddings between epochs. For 3D Gaussian noise E[|noise|] is about 1.596 sigma, so sigma = drift / 1.596.
- **Conditions:** prescribed; noise_late (sigma 0.0031, drift of epoch 28 to 29); noise_mid (0.0501, epoch 2 to 3); noise_early (0.8961, epoch 0 to 1); noise_schedule (sigma follows the measured drift schedule); correlated_mid (one displacement per epoch, 0.0501); correlated_schedule; free (MLP 5 to 3).
- Noise is applied only in training; evaluation uses the clean prescribed latent.

## Files
- `code/noise_control.py`: script, 3 seeds, 8 conditions
- `results/noise_control_results.json`: full results

## Facts
Ф41i, Ф42i, Ф43i, Ф44i

## Status
Summary table: noise side stands (Ф87); free side not interpretable (Ф82).

- Ф41i: no correction recorded in the registry; stands as recorded
- Ф42i: no correction recorded in the registry; stands as recorded
- Ф43i: Ф82: the free 222x side is not interpretable; the noise side stands.
- Ф44i: no correction recorded in the registry; stands as recorded
