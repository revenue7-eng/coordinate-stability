# Coordinate Stability in Learned Representation Spaces

**Research question:** Does a learning system need a stable coordinate system to learn effectively — and what happens when it doesn't have one?

## Context

Joint-Embedding Predictive Architectures (JEPA) learn by predicting masked representations from visible context. The standard approach: let the encoder discover its own representation space during training. We investigate what happens when the coordinate system is fixed before training instead.

## What we observe

Across 29 experiments on multiple environments and modalities:

1. **Fixed coordinates consistently outperform learned ones** in the low-data regime (5–222× improvement), regardless of whether the axes carry semantic meaning. Any frozen, normalized basis in the task-relevant subspace works equally well (random ≈ prescribed, ratio 0.97×).

2. **The free encoder's coordinate system drifts** — it restructures so rapidly in early training that a linear decoder trained at epoch 0 produces outputs worse than random at epoch 1 (R² < −62). Standard geometric metrics (rank, isotropy) remain healthy throughout.

3. **Drift is not generic noise.** Matched-amplitude i.i.d. noise is 4× more destructive than actual drift; matched-amplitude correlated noise (constant shift) is 167× less destructive. The free encoder's instability is a structured, data-dependent deformation between these extremes.

4. **Standard remedies don't solve it.** Reducing encoder learning rate by 100× leaves a 62× gap. Extra predictor updates, EMA target encoders, and PCA alignment all fail. The instability is structural, not an optimization artifact.

5. **Fixed coordinates have a ceiling.** With sufficient data (500 episodes), the free encoder surpasses prescribed by six orders of magnitude. Coordinate fixation provides sample efficiency, not absolute superiority.

## Papers

- **Paper 1:** "The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization" — [prescribed-axes repo](https://github.com/revenue7-eng/prescribed-axes)
- **Paper 2:** "Semantic Drift, Not Rank Collapse: Why Stable Coordinates Matter for Learning in Joint-Embedding Spaces" — [prescribed-axes-drift repo](https://github.com/revenue7-eng/prescribed-axes-drift)

## Repository structure

Each experiment follows a standard protocol:

```
E{NN}_{name}/
  code/        — scripts and notebooks
  results/     — JSON data, figures
  README.md    — goal, conditions, metrics, hypothesis, results
```

- [EXPERIMENTS.md](EXPERIMENTS.md) — Full registry of all experiments with parameters and key results
- [EVIDENCE.md](EVIDENCE.md) — Experimentally verified facts, hypotheses with status (confirmed/refuted/open), and contradictions

## Experiments (29)

| ID | Name | Environment | Key result |
|---|---|---|---|
| E01 | Speech JEPA | LibriSpeech | +18–20pp entropy |
| E02 | LeWM State | Push-T | 38× prescribed advantage |
| E03 | LeWM Pixel | Push-T pixels | 14.8×, 37× fewer params |
| E04 | Shov-JEPA Vision | Rico UI | +5% accuracy |
| E05 | Controls | Push-T | random ≈ prescribed |
| E06 | Covariance + Drift | Push-T | rank 2.99 → still 222× worse |
| E07 | Freeze test | Push-T | freeze@1 +20% |
| E08 | Random fixed encoder | Push-T | 17× stability effect |
| E09 | Aligned-but-drifting | Push-T | aligned ≈ free |
| E10 | LR sweep + EMA | Push-T | prescribed wins at every LR |
| E11 | Rico drift | Rico UI | cross-modal drift confirmation |
| E12–E14 | Dimension sweep | Push-T | prescribed wins dim 1–11 |
| E15 | Simple pendulum | Pendulum | free wins (boundary condition) |
| E16 | Double pendulum | Double pendulum | normalization resolves |
| E17 | Fragility test | Push-T | noise axis: 1106× degradation |
| E18 | MLP decoder transfer | Push-T | two-phase drift model |
| E19 | Update ratio + diffLR | Push-T | 62× gap remains at 100× slower |
| E20 | PCA canonicalization | Push-T | PCA worsens transfer |
| E21–E22 | Confound tests | Push-T | optimizer/SIGReg confounds absent |
| E23 | Random 3D vs 5D | Push-T | subspace selection critical |
| E24–E26 | Dimensionality scaling | Push-T | 3D→16D: gap persists, drift scales |
| E27 | Drift correlation | Push-T | Pearson = 0.95 |
| E28 | Full dim sweep | Push-T | NO crossover at any dimension |
| E29 | Noise control | Push-T | drift ≠ noise ≠ shift |

## Current status

- Paper 1: published to GitHub, arxiv submission pending
- Paper 2: in revision (incorporating E18–E29 results)
- EB-JEPA (planning task at realistic scale): running

## Author

Andrey Lazarev — Independent Researcher
lazarev@tactiqedge.com
