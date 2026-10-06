# Coordinate Stability in Learned Representation Spaces

**Research question:** does a representation whose coordinate system is fixed before training (prescribed axes) give a better world model than one whose coordinates are learned, and what does the movement of learned coordinates cost a downstream module?

## Where the programme stands

The early results (E01 to E29) reported prescribed axes ahead of learned encoders by 5x to 1820x. Those ratios do not survive the audit. Two measurement defects explain them: the free encoder read the raw state through a first layer that discards its scale (Ф76), and each encoder was scored on its own latent, so ratios across different latent geometries are not comparable (Ф77). The status of every older comparison is recorded in EVIDENCE Ф80, Ф82 and Ф87.

What stands, each in the scope stated in EVIDENCE:

- On a common target with equally scaled input, a fixed and a learned encoder are not distinguishable at dim 5 (E44, E45; class O4 PROVISIONAL; Ф78, Ф79).
- On fully observed synthetic Push-T the standardised state beats a frozen JEPA latent of the same state, by more at low data (Ф84, Ф85). The latent is complete but nonlinearly laid out (Ф86). There the prescribed arm is the state itself, so this measures JEPA self-supervision, not fixed coordinates of a learned representation (Н5). E50, built to separate fixation from geometry, is void; its exploratory numbers point to geometry and show an encoder trained end to end on the target ahead of the fixed basis.
- Among fixed latents in the same units the comparisons stand (Ф13 aside, see Ф87). random_fixed matching prescribed (Ф5, Ф31, Ф39) is a rotation of the same coordinates, so it does not show that the meaning of the axes is irrelevant.
- The coordinates of a learned encoder move strongly in the first epoch and less later, measured by decoder transfer R2 (Ф10, Ф24, Ф25, Ф27, Ф35).
- Noise injected into a fixed latent during training harms the downstream module, and the harm depends on its structure (Ф41i, Ф42i, Ф44i).
- The critical-window results (E30 to E32, E38, E39) score the predictor on the encoder's own latent; whether they measure damage or a growing spread of the latent is open (Ф87).
- Among frozen random initialisations, linear readability of the state orders downstream quality on an external target (Ф75).
- Two Rooms planning: E35 compares oracle-state input with pixels, both trained, on one seed (Ф58, Ф81); it does not test fixed axes.
- LLM residual streams (E33): the leading principal components are a last-token artefact (Ф47 to Ф55).

## Papers

- Paper 1: "The Space Matters More Than the Loss", [prescribed-axes](https://github.com/revenue7-eng/prescribed-axes)
- Paper 2: "Semantic Drift, Not Rank Collapse", [prescribed-axes-drift](https://github.com/revenue7-eng/prescribed-axes-drift)

Both predate the audit. Their quantitative prescribed-versus-free claims rest on own-latent ratios (Ф77, Ф82).

## Repository structure

```
E{NN}_{name}/
  code/        scripts and notebooks
  results/     JSON data, logs, figures
  README.md    goal, conditions, metrics, status
```

- [EXPERIMENTS.md](EXPERIMENTS.md): registry of all experiments, pre-registrations and a summary table with the current status of each
- [EVIDENCE.md](EVIDENCE.md): facts (Ф), observations (Н), hypotheses (Г), contradictions (П); the registry is authoritative over this README

Model checkpoints (`*.pt`, `*.pth`, `*.pth.tar`) are not tracked; their MD5 sums are recorded in the experiment READMEs where they exist.

## Author

Andrey Lazarev, Independent Researcher
lazarev@tactiqedge.com
