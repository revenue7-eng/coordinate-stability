# PAPER 1 — SKELETON (rev 4, 2026-09-14)

Scope: variant B, negative claim, opening-window line (E39 → E40 → E41 → E42).
The Г25 / prescribed-vs-free line is excluded and becomes paper 2.

`[Фnn]` = registered in EVIDENCE.md. `[NEEDS]` = blocks writing until resolved.
`[rev4]` marks what changed from rev 3.

**Rev 4 correction.** Revisions 1 to 3 described the setting as EB-JEPA on Two
Rooms with MPPI planning. That is the Г25 line, not this one. This line runs on
gym-pusht with a frozen encoder and a trained predictor head; there is no planner
in it. The numbers (Ф64 to Ф68) were computed over this line's own result files
and are unaffected; every sentence describing what was measured had to be
rewritten. `[verified: e40_lib.py:34-56; grep for MPPI/plan over E39, E40, E41
code returns nothing]`

Descriptive statements below now carry tags like the numerical ones. The three
errors caught this session (a claimed agreement with RankMe, a spectrum refusal,
the Two Rooms setting) were all untagged background rather than probed claims.

---

## Working title candidates `[rev4: the planning-flavoured title is dropped]`

1. Frozen random bases differ, and nothing cheap tells you which one you got
2. The spread between frozen initialisations belongs to the encoder and resists
   explanation

## Abstract — three sentences

1. Frozen random initialisations of the same encoder architecture differ
   substantially in the validation loss a downstream predictor reaches on top of
   them. `[rev4: was "downstream planning performance"]`
2. The difference is a property of the encoder, not of the encoder-head pair
   (variance share 0.993). `[Ф65]`
3. Linear informativeness of the representation does not order them, and neither
   does effective rank. `[Ф64] [Ф67]`

H1 decided: the two-tier shape of the spread stays in section 5, not the
abstract. It rests on two observations out of ten and E42 can move it. E42
therefore keeps its correlational target and the n=30 sizing stands.

## 1. Introduction

- The question: a frozen random encoder is a fixed coordinate system. Do such
  systems differ, and if so by what?
- Origin of the line, and it is a usable narrative: E39 left an observation that
  looked like a trend, the initialisation carrying the most linearly extractable
  information about the true state also gave the worst result. It could not be
  read either way, because initialisation and data sample varied together. E40
  fixes the data seed at 42 and varies only the initialisation. `[verified:
  E40_init_sweep/README.md]`
- Why it matters: initialisation is chosen by default, not by measurement.
- Not claimed here: nothing about trained encoders, nothing about prescribed axes
  (paper 2), nothing about planning — this line contains no planner. `[rev4]`

## 2. Related work

The niche none of these covers: variation between seeds of one architecture at
fixed hyperparameters, in an encoder that never trains.

- **Frozen random features.** Randomly initialised convolutional features are
  strong without training (Jarrett et al.; Saxe et al.). Closest neighbour:
  frozen random CNN extractors in deep RL (arXiv 2607.26059, 2026), reporting
  that seed variance exceeds width effects. Their question is emergent sparsity.
  `[NEEDS: read in full; if they report per-seed numbers, put them beside ours]`
- **Counter-evidence worth citing.** Frozen-transformer time-series work reports
  negligible seed variance from freezing (arXiv 2508.18130). Our claim is
  regime-specific, not universal. Saying so strengthens it.
- **Spectral criteria.** RankMe (Garrido, Balestriero, Najman, LeCun, ICML 2023)
  proposes effective rank of embeddings as a label-free predictor of downstream
  performance in joint-embedding SSL. Disjoint from us on four axes: untrained vs
  trained encoders, seeds vs hyperparameters, 3-dimensional output vs
  high-dimensional embeddings, predictor loss vs linear probing. We neither
  confirm nor contradict it. `[rev2 claimed agreement in the bulk; that was an
  artifact of the biased subset 1..8, withdrawn, see Ф67]`
- **Cheap properties of untrained networks.** Zero-cost proxies in NAS estimate
  trained accuracy from an untrained network on one minibatch; simple baselines
  such as parameter count stay competitive and proxies transfer poorly across
  benchmarks (grad-norm 0.58 on NAS-Bench-201, -0.21 on NAS-Bench-NLP). All of it
  compares architectures, not seeds of one.
- **Seed variance generally.** Documented for trained networks. Frozen encoders
  are the gap.

## 3. Setup `[rev4: rewritten]`

- Data: gym-pusht, `gym.make("gym_pusht/PushT-v0", obs_type="state")`, episodes
  collected by stepping the environment with sampled actions. `[verified:
  e40_lib.py:34-56]`
- Objective: SIGReg-regularised joint embedding; the predictor head trains, the
  encoder does not. `[verified: e40_lib.py SIGReg class]`
- Frozen regime: `freeze_frac=0`, the encoder is frozen before the first
  optimiser step and never trains. `[verified: e40_lib.py:120-125, 136-138]`
- Metric: `best_vp` is the minimum validation loss over epochs; lower is better.
  `[verified: e41_lib.py:154,157]`
- No planner, no planning success rate anywhere in this line. `[verified: grep
  MPPI/plan over E39, E40, E41 code]`
- E40: ten initialisations at data seed 42. E41: crossed 8 encoder x 5 head
  seeds, 40 cells; encoder seeds 1..8 bit-identical to E40 inits 1..8. `[Ф65]`
- Code lineage: `e40_lib.py` copies `e39_lib.py` copies `e32_lib.py`; each
  experiment owns its library so registered results stay reproducible as
  recorded, and patch scripts assert each substitution matches exactly once.
  `[verified: E40_init_sweep/README.md]` `[rev4: this closes H6]`

## 4. The spread belongs to the encoder

- Variance share: encoder 0.993, head 0.002, residual 0.005; F_enc=1021.2 (df
  7/28), F_head=4.92. `[Ф65]`
- Stable across statistic and scale. `[Ф65]`
- The encoder level is recoverable from one run: +0.9951 Pearson. `[Ф66]`
- Consequence: one head per initialisation suffices; no crossed design needed.
  `[Ф66]`

State that the head effect is statistically detectable and negligible in size, or
a reader who checks F_head will think 0.002 was rounded away.

## 5. The shape of the spread

- Ten initialisations: 7.3086x, CV 0.556. Leave-one-out moves it only at init 7
  (3.9544) and init 9 (4.7191); no third sensitive point. Core without both:
  2.5533x, CV 0.256. `[Ф68]`
- Eight head-averaged levels: 3.873x, 2.001x without e7; on mean_last3 3.576x and
  1.858x. `[Ф68]`
- Not selection along the trajectory: best_vp equals final_vp in 27 of 40 cells;
  the ratio loses 7.7% on the non-selective statistic. `[Ф68]`
- The upper extreme is reproducible across all five heads. `[Ф68]`

Formulation: the bulk spreads about 2x at CV ~0.24, robust to dropping any single
point; rare initialisations depart by a further factor of ~3. Two numbers, not
one. Same paragraph carries the limit: the tail rests on two observations of ten.

## 6. What does not explain it

Two refusals, not three. The "spectrum was withdrawn" line carried in revisions 1
and 2 had no source in any handoff, EVIDENCE entry or EXPERIMENTS metric for this
line and was removed. On a 3-dimensional output the whole spectrum is three
numbers and effective rank is their standard summary, which we did test.
`[verified: grep -ln "spectr\|eigen" HANDOFF_*.md returns nothing]`

- Linear informativeness: corr(best_vp, R2_readout) = +0.060 over ten, while
  R2_readout itself spans 0.2385..0.5156. `[Ф64]`
- Effective rank: -0.2871 over ten, -0.6994 on the grid subset, +0.3919 without
  init 7, and -0.1843 / -0.3095 on the core with both extremes removed, where the
  resolution limit at n=8 is |rho|=0.79. No signal in the bulk; the grid-subset
  figure comes from including init 7 and excluding init 9. `[Ф67]`
- eff_rank does vary (1.61 to 2.70 of maximum 3), so the non-result is not a
  constant-predictor artefact. `[Ф63, measured in E39 — see the comparability
  caveat in limits]` `[rev4]`

Strongest paragraph available: the two points that set the direction of the
eff_rank correlation are the same two that set the magnitude of the spread. One
property of the data seen twice. State it as an observation; there is no
mechanism for why low rank accompanies extreme outcomes in both directions, and
it rests on two points. `[Ф68 CONNECTION]`

## 7. Methodological note

- Changing the set of points and the aggregation at once manufactures an apparent
  mechanism. `[Ф67]`
- Rule: re-run on the original composition before explaining a difference by a
  mechanism.
- Power arithmetic for readers sizing their own sweep: n=10 resolves |rho| from
  0.785, n=20 from 0.591, n=30 from 0.492; rho=0.4 needs n=47. `[verified: Fisher
  z, 2026-09-14]`

## 8. Limits

- Single data sample (seed 42). `[NEEDS: H2 — 8-10 encoders on a second data
  seed, no crossed design needed per Ф66]`
- `[rev4]` E39 and E40 are not directly comparable at step 0: E40 reseeds the
  stream before building the model and restores it immediately after, so E40 runs
  are comparable among themselves but not against E39's step 0, where the stream
  was never touched. Any figure carried from E39 (Ф63) must say so. `[verified:
  E40_init_sweep/README.md design note]`
- Frozen regime only; whether the ordering survives unfreezing is untested.
- 3-dimensional encoder output: effective rank has little resolution there, so
  the eff_rank non-result does not transfer to high-dimensional embeddings and is
  not evidence about RankMe. `[H7]`
- One environment, one architecture; ten and eight initialisations.
- No EMA-teacher variant.

## 9. The open question

- Г27 with its premise measured: the spread belongs to the encoder (0.993), so
  interaction with the downstream module is no longer available as an
  explanation. `[Ф65]`
- Refined: the target is not a property that orders frozen bases in general, but
  one that explains the tails. The bulk is where nothing cheap shows a signal and
  where a signal would matter least.
- If none is found, the property that orders the bases is not cheap. A result,
  not a gap.

## 10. What E42 adds

As registered: ~30 initialisations, one head each, candidate predictors computed
before training, n=30 resolving |rho| from 0.49. `[EXPERIMENTS.md E42]` Sizing
unchanged. E42 also measures the distribution of levels at n=30, which is what
the section-5 limit needs, and that comes free with the sweep.

---

## Holes

- **H1.** CLOSED. Two-tier shape in section 5, not the abstract.
- **H2.** OPEN. Second data seed; fold into the E42 campaign.
- **H3.** CLOSED. `best_vp` defined.
- **H4.** CLOSED. Ф68 committed (a236ae6).
- **H5.** CLOSED by removal. The spectrum refusal did not exist.
- **H6.** CLOSED `[rev4]`. Code lineage and per-experiment libraries documented
  in E40's README; no upstream vendoring question arises for this line, which was
  a Г25 concern misattributed here.
- **H7.** OPEN as a limit. Optional strengthening: compute eff_rank before the
  3-d projection, which would give either a real second refusal or a real lead.
- **H8** `[rev4, new]`. Every descriptive claim in this document now carries a
  probe. Any sentence added later about the setting, the environment, the metric
  or the lineage must carry one too. Three errors this session came from untagged
  background.

## Not in this paper

Г25, prescribed vs free, E35, prescribed_3, Ф56, Ф58, Two Rooms, MPPI — paper 2.
Г17 / epiplexity — unrelated line.
