# Prescribed Axes: Facts and Hypotheses

Andrey Lazarev | Independent Researcher

Last updated: 20 August 2026 — merge of the April branch (LLM → E33, EB-JEPA → E34/E35, Ф47–Ф55, Н1–Н4, Г19–Г25) with the July branch (Ф45, Ф46, Г16–Г18). April E- and Г-numbers were reassigned, July ones kept. See MAPPING TABLES at the end of the file.

Protocol:
- **Facts (Ф)** — experimentally verified results, ours or from sources we have studied. For ours: code, environment, parameters and seeds are given. For external ones: the source is given (book, paper with DOI/arXiv, page or section). Admitting an external fact to the registry is a curatorial decision: it counts as established once independent work confirms it, or once it is adopted as a working basis for the current programme.
- **Observations (Н)** — single-seed or underpowered results. Not facts under the programme protocol. Recorded for signal and audit, not as support for hypotheses.
- **Hypotheses (Г)** — carry a status (confirmed / refuted / open).

---

## FACTS

### Environment: Push-T (3 degrees of freedom: x_block, y_block, θ_block)

**Ф1. Prescribed (3D) vs free (5D→3D): prescribed better by 38×**
- Prescribed: 0.004, free: 0.157 (val loss)
- 3 seeds, 50 epochs, 200 episodes
- Paper 1, Experiment 3 (LeWM State)

**Ф2. Prescribed (3D, 20K params) vs free CNN (96×96 pixels, 744K params): prescribed better by 14.8×**
- The CNN plateaus at epoch 7 of 50
- Paper 1, Experiment 4 (LeWM Pixel)

**Ф3. Speech JEPA: prescribed (frozen cluster anchors) beats free by +18–20pp entropy**
- 2×2 factorial: {GMM, k-means} × {soft, hard}
- Every prescribed condition beats free
- Dominant factor: frozen structure, not the clustering method
- Pilot study; metric is entropy (codebook utilization)
- Paper 1, Experiment 1

**Ф4. Shov-JEPA (vision): 3 prescribed axes beat 64 free ones, 72.5% vs 67.5%**
- Rico dataset, 398 UI screenshots
- Pilot (398 samples, single seed, +5%)
- Paper 1, Experiment 2

**Ф5. Random fixed axes (3D) ≈ prescribed**
- Random fixed: 0.61× of prescribed (slightly better) at 200 ep
- Random fixed: 1.00× at 500 ep
- Free 3D same input: 4.47× worse than prescribed at 200 ep
- Fixedness matters more than the semantics of the axes
- Source of 0.61×: random_axes_control, 200 ep, 30 epochs, 3 seeds, no SIGReg (Ф39)
- Source of 1.00×: random_axes_control, 500 ep, 50 epochs, 9 runs (Ф39)
- Paper 1, Section 4.6 + random_axes_control/RESULTS.md

**Ф6. Equal-input control: free with the same input (x, y, θ) is 7.6× worse than prescribed**
- Prescribed: 0.000472, free 3D same input: 0.003570
- The advantage comes from fixedness, not from access to information
- Paper 1, reviewer response

**Ф7. SIGReg on prescribed: effect 0.6%. On free: effect 1.9×**
- Removing SIGReg *improves* the free encoder (0.006 vs 0.012)
- SIGReg treats what prescribed prevents
- Paper 1, reviewer response

**Ф8. SIGReg can harm the free encoder**
- Free without SIGReg: 0.037, free with SIGReg: 0.156 (4.2× worse)
- SIGReg forces isotropy; the task is anisotropic
- Eigenvalues, prescribed: [0.098, 0.078, 0.064] — reflects the structure of the task
- Eigenvalues, free+SIGReg: [0.98, 0.93, 0.85] — artificial isotropy
- Paper 2, Section 6.2

**Ф9. The free encoder has full rank (2.99/3) and isotropy (0.86) — and loses to prescribed by 233×**
- Prescribed: rank 2.91, isotropy 0.66
- Rank collapse is not the cause of the degradation
- Paper 2, Section 3

**Ф10. The free encoder drifts: R² transfer < −62 after a single epoch**
- Seed 42: −16.9, seed 123: −62.2, seed 777: −25.4
- A linear decoder from epoch t is catastrophically wrong at epoch t+1
- 80% of the drift is structural (after Procrustes alignment)
- By epoch 2→3 R² recovers to 0.73–0.76
- Paper 2, Section 4.2

**Ф11. Freeze@1 improves free by 20%**
- Free unfrozen: 0.081, freeze@1: 0.065
- Freeze@2: +2.9%, freeze@10: −1.1% (neutral)
- Causal evidence: stabilizing the encoder helps
- But freeze@1 (0.065) is still 25× worse than prescribed (0.0025)
- Stability alone is not sufficient
- Paper 2, Section 5.1

**Ф12. A random fixed encoder beats free by 17×**
- Random fixed: 0.000476, free: 0.008282
- Random fixed = frozen random orthogonal projection, zero semantic content
- Stability without alignment already gives an order-of-magnitude advantage
- Paper 2, Section 5.4

**Ф13. Prescribed beats random fixed by 13×**
- Prescribed: 0.000036, random fixed: 0.000476
- Alignment adds a further advantage on top of stability
- Paper 2, Section 5.4

**Ф14. Rotated prescribed ≈ prescribed (1.09×)**
- Interpretability of the axes does not matter
- What matters: fixedness + the right subspace
- Paper 2, Section 5.4

**Ф15. Aligned-but-drifting ≈ free (or worse)**
- Aligned-drifting linear: 0.012849 (1.55× worse than free at 0.008282)
- Aligned-drifting MLP: 0.008380 (≈ free)
- Encoder initialized at the ideal coordinates → drift allowed → the advantage is lost entirely
- Alignment without stability is useless
- Paper 2, Section 5.5

**Ф16. Stability × alignment are not independent but hierarchical**
- 2×2 factorial:
  - Stable+aligned (prescribed): 0.000039
  - Stable+unaligned (random fixed): 0.000473
  - Unstable+aligned (aligned-drifting): 0.012849
  - Unstable+unaligned (free): 0.008282
- Stability effect among aligned: 330×
- Stability effect among unaligned: 17.5×
- A 19× difference → strong interaction
- Stability is the prerequisite. Alignment adds 12× only when stability is present
- Paper 2, Section 5.5

**Ф17. Prescribed 11D is 20× worse than prescribed 3D, but 42× better than free 11D**
- prescribed_3: 0.000036, prescribed_11: 0.000732 (20× worse than prescribed_3)
- free_11: 0.030509 (42× worse than prescribed_11)
- Original data (E12, a different setup): free_11 (0.000060) beat prescribed_11 (0.000381) by 6×
- Source of the discrepancy: E12 used a fixed predictor hidden=128; E28 uses max(128, dim*8)
- With the right predictor capacity, prescribed_11 wins over free_11
- Redundant prescribed axes degrade prescribed (20× vs prescribed_3), but not enough for free to win
- Verified: p2_dim_sweep_results.json (E28)
- 200 episodes, 30 epochs, 3 seeds

**Ф18. ~~Sweep dim 1→15: prescribed wins only at dim ≤ 3~~ REFUTED**
- The original data (100 ep, 20 epochs, 2 seeds) showed a crossover at dim=4
- **A full rerun (200 ep, 30 epochs, 3 seeds) refuted the crossover:**
- dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
- Prescribed wins at ALL dimensions 1–11
- The gap shrinks monotonically as dim grows (228× → 42×) but does not vanish
- dim=2 is anomalously high (1820×) — prescribed [x_b, y_b] is a perfect match
- dim=5 agrees with Tier 3 E25 (66.3× vs 66.2× — full reproducibility)
- The original result was an artefact of an underpowered setup
- Verified: p2_dim_sweep_results.json, per-seed breakdown
- Experiment E28 (15.04.2026)

**Ф19. Prescribed does not win at any dim (1–5)**
- Free is better at every dimension
- Prescribed and free receive the same 2D input
- 100 episodes, 20 epochs, 3 seeds
- Experiment, 15.04.2026

**Ф21. An unpredictable axis destroys prescribed catastrophically (1106×)**
- prescribed_4_noise (x, y, θ + frozen random): 0.021534
- prescribed_3 (x, y, θ): 0.000019
- The predictor is required to predict the unpredictable → total failure
- 200 episodes, 30 epochs, 3 seeds
- Experiment 15.04.2026 (fragility test)

**Ф22. Axes from the same subspace do less damage than axes from another**
- prescribed_4_sin (x, y, θ + sinθ): 0.000093 (4.8× vs p3) — redundant, same subspace
- prescribed_4_dist (x, y, θ + d_agent_block): 0.000160 (8.2× vs p3) — redundant, cross-subspace
- prescribed_4_agent (x, y, θ + agent_x): 0.000154 (7.9× vs p3) — independent, cross-subspace
- sin(θ) is roughly twice as good as dist and agent_x
- The subspace matters: an axis from the same subspace (the block) is less harmful
- 200 episodes, 30 epochs, 3 seeds
- Experiment 15.04.2026 (fragility test)

**Ф23. Redundancy vs independence of the 4th axis: the effect is the same cross-subspace**
- prescribed_4_dist (redundant, cross-subspace): 0.000160 (8.2×)
- prescribed_4_agent (independent, cross-subspace): 0.000154 (7.9×)
- Under 5% apart → the issue is not redundancy as such but the widening of the subspace
- 200 episodes, 30 epochs, 3 seeds
- Experiment 15.04.2026 (fragility test)

### Environment: double pendulum (4 degrees of freedom: θ1, ω1, θ2, ω2)

**Ф20. ~~Prescribed wins only at dim=1~~ REFUTED — normalization settles it**
- Original (no normalization): prescribed loses to free at dim ≥ 2
- **With [0,1] normalization prescribed beats free at ALL dims 1–8:**
  - dim=1: 9.8×, dim=2: 16.8×, dim=3: 12.3×, dim=4: 19.1×
- Normalization gives a 37–166× improvement to the prescribed encoder
- Prescribed raw: 0.05–0.21, prescribed normalized: 0.001–0.003, free: 0.014–0.035
- Prescribed and free receive the same 4D input — no selection
- **П1 CLOSED: the Push-T vs pendulum difference was normalization**
- 200 episodes, 30 epochs, 3 seeds, per-seed verified
- Experiment E16 fix (15.04.2026)
- Data: fix_e16_results.json

### Tier 1 tests (critical tests of hypotheses, 15.04.2026)

**Ф24. Early drift (ep 0→1) destroys information — even an MLP decoder breaks**
- Epoch 0→1: MLP decoder R² transfer = −283 (mean over 3 seeds)
- Linear decoder R² transfer = −71 (mean over 3 seeds)
- MLP self R²: 0.80–0.95 (well fitted at epoch t)
- The MLP decoder CANNOT recover the ground truth from epoch t+1 embeddings
- The information is genuinely destroyed, not merely unreadable by a linear map
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 1 / T1

**Ф25. Late drift (ep 2+) preserves information in a non-linearly deformed form**
- Epoch 5→6: MLP decoder R² transfer = 0.79, linear = 0.69
- Epoch 29→30: MLP decoder R² transfer = 0.81, linear = 0.69
- MLP advantage (late epochs): +0.12 R² on average
- The information is present but not linearly readable — the MLP recovers it
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 1 / T1

**Ф26. Differential LR (encoder 100× slower) closes 72% of the gap, but 62× remains**
- Prescribed: 0.000037
- Free K=1: 0.008282 (222×)
- Free diffLR 100× (enc LR=3e-6): 0.002303 (62×)
- Extra predictor steps (K=3): 0.010399 — WORSE than K=1
- Extra predictor steps (K=5): 0.006377 (171×)
- diffLR 10× (enc LR=3e-5): 0.007612 (204×) — weak effect
- The problem is NOT purely optimization lag
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 1 / T2

**Ф27. PCA canonicalization does not help — the drift is non-linear**
- PCA degrades R² transfer at most epochs
- Epoch 0→1: raw = −71.4, PCA = −69.1 (negligible difference)
- Epoch 2→3: raw = 0.678, PCA = −0.376 (PCA WORSE)
- Epoch 29→30: raw = 0.690, PCA = 0.298 (PCA WORSE)
- Drift is a non-linear deformation, not rotation/scaling
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 1 / T3

**Ф28. Drift rate correlates with val loss: Pearson = 0.95**
- The correlation is dominated by epoch 0→1 (an outlier)
- Spearman = 0.51 (weaker — the dependence is non-linear)
- Two regimes: a "catastrophe zone" (drift > 0.3) and a "saturation zone" (drift < 0.1)
- After epoch 3 drift stops being the bottleneck and alignment dominates
- The R² ceiling of the free encoder is ≈ 0.75 (linear); prescribed ≈ 1.0
- Data from all_results.json (gym-pusht), 3 seeds, 30 epochs
- Tier 1 / T8

### Tier 2 tests (confound tests, 15.04.2026)

**Ф29. SIGReg stabilizes aligned-drifting linear (it does not destroy it)**
- Aligned-linear with SIGReg: 0.013314 (stable across seeds: 0.013, 0.016, 0.011)
- Aligned-linear without SIGReg: 0.470604 (unstable: 0.016, 1.393, 0.003)
- Seed 123 without SIGReg: 1.39 — catastrophic divergence
- SIGReg prevents divergence of the linear encoder
- But neither with SIGReg (356×) nor without does aligned-drifting come near prescribed
- Free with SIGReg: 0.008282, free without: 0.007345 (SIGReg mildly harms free — consistent with Ф8)
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 2 / T4

**Ф30. The optimizer reset in the freeze test is NOT a confound**
- freeze@1, new optimizer: 0.008785 (−6.1% vs unfrozen)
- freeze@1, keep state: 0.008701 (−5.1% vs unfrozen)
- Difference: 1.0%
- freeze@3, new optimizer: 0.007108 (+14.2%)
- freeze@3, keep state: 0.006898 (+16.7%)
- Difference: 2.9%
- Preserving optimizer state does not change the freeze-test result
- HOWEVER: freeze@1 on synthetic data makes things worse (−6%) rather than better (+20% on gym-pusht)
- The freeze effect depends on the data, but there is no optimizer confound
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 2 / T5

**Ф31. Random fixed 3D (from block coordinates) ≈ prescribed = rotated prescribed**
- prescribed: 0.000037
- rotated_prescribed: 0.000039 (1.05×)
- random_fixed_3d: 0.000036 (0.97×)
- All three are ≈ equal
- Any stable orthogonal basis in the right subspace equals prescribed
- Alignment of the axes within the subspace does not matter
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 2 / T7

**Ф32. Random fixed 5D (from all coordinates, unnormalized) blows up**
- random_fixed_5d: 376,053 (mean; per seed: 977K, 25K, 125K)
- An unnormalized projection from the full space is unstable
- The original "17× stability advantage" from Paper 2 (Ф12) was obtained on particular seeds with a different random_fixed implementation
- Tier 2 / T7

### Tier 3 tests (generalization, 15.04.2026)

**Ф33. The prescribed vs free gap reproduces in 3D, 5D and 16D**
- 3D: prescribed 0.000050, free 0.008503, gap 169×
- 5D: prescribed 0.000354, free 0.023431, gap 66×
- 16D: prescribed 0.000747, free 0.037105, gap 50×
- The gap narrows as dimension grows but stays order-of-magnitude
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 3 / T9a, T9b

**Ф34. Random fixed ≈ prescribed in 5D (0.92×), starts to fall behind in 16D (1.53×)**
- 5D: random_fixed = 0.000324, prescribed = 0.000354 (random is slightly better)
- 16D: random_fixed = 0.001142, prescribed = 0.000747 (prescribed better by 1.53×)
- In 5D (linear coordinates) alignment within the subspace does not matter
- In 16D (non-linear features) alignment starts to matter
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 3 / T9a, T9b

**Ф35. Drift grows with dimension**
- 3D: drift_01 = 1.53, R² transfer = −70
- 5D: drift_01 = 1.91, R² transfer = −65
- 16D: drift_01 = 3.58, R² transfer = −596
- R² transfer in 16D is 8.5× worse than in 3D
- More dimensions → more "flat directions" → stronger drift
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 3 / T9a, T9b

**Ф36. 5D prescribed (all coordinates, no selection) works — gap 66×**
- prescribed_5d = normalize(all 5 coordinates), free_5d = MLP 5→5
- Prescribed does NOT pick a subspace — it takes everything
- Gap 66× — smaller than 3D (169×), still order-of-magnitude
- Contradicts Ф19 (pendulum), where prescribed on the full input does not work
- The difference: Push-T 5D prescribed normalizes, the pendulum does not (?)
- Or: Push-T contains "extra" coordinates (the agent), the pendulum does not
- 200 episodes, 30 epochs, 3 seeds, synthetic
- Tier 3 / T9a

### Verified from the Paper 1 archive (15.04.2026)

**Ф37. Gauge fixing the free encoder does not help (1.08× ≈ free)**
- prescribed: 0.000472, free: 0.011614, gauge_fixed_free: 0.012581
- linear_free: 16009 (blow-up)
- Gauge fixing (pinning the symmetry through training) does not solve drift
- Data seed 42, training seeds [42, 123, 777], 50 epochs, synthetic
- Date: 12.04.2026
- Source: 1.rar/gauge_fix_results/results.json
- Was not previously included in the fact registry

**Ф38. At 500 episodes the free encoder beats prescribed by 695,000×**
- prescribed: 8.513×10⁻⁴ (3 seeds, std 8.4×10⁻⁷)
- random_fixed: 8.514×10⁻⁴ (9 runs: 3 rotation seeds × 3 training seeds, std 2.1×10⁻⁶)
- free_3d: 1.225×10⁻⁹ (3 seeds, std 3.7×10⁻¹⁰)
- free_5d: 2.916×10⁻⁹ (3 seeds, std 7.8×10⁻¹⁰)
- Fixed encoders (prescribed and random) plateau at ~8.5×10⁻⁴ — irreducible error from min-max normalization
- The free encoder converges to ~10⁻⁹ (effectively zero)
- **Prescribed axes buy sample efficiency, not an absolute advantage**
- 500 episodes, 50 epochs, synthetic, no SIGReg
- Verified from JSON: exp5_random_axes/all_results.json (18 runs)
- Paper 1, random_axes_control/RESULTS.md

**Ф39. random_fixed ≈ prescribed at both data scales**
- 200 ep: random 0.61× (slightly better than prescribed) — 3 seeds, 30 epochs
- 500 ep: random 1.00× (identical) — 9 runs vs 3 runs
- Axis semantics are secondary at any amount of data
- Verified from JSON: random_fixed_results/results.json (200ep), all_results_500ep.json (500ep)
- Paper 1, random_axes_control/RESULTS.md

**Ф40. Isotropic normalization (zero-mean, unit-variance) degrades by 15×**
- prescribed_iso: 0.010250 vs prescribed: 0.000673 (15.2×)
- random_fixed_iso: 0.007019 vs random_fixed: 0.000408 (17.2×)
- Min-max [0,1] normalization is critical; standardization hurts
- 200 episodes, 20 epochs, synthetic
- Paper 1, random_axes_control/RESULTS.md

### Noise control (E29, 17.04.2026)

**Ф41i. i.i.d. noise at drift-matched amplitude (ep 0→1, σ≈0.90) destroys prescribed by 851×**
- noise_early: 0.031781, prescribed: 0.000037
- Free encoder: 0.008282 (222×) — 4× better than i.i.d. noise
- Drift does not reduce to a random perturbation
- 3 seeds, 200 episodes, 30 epochs, synthetic
- E29

**Ф42i. Correlated noise (a constant shift per epoch) barely harms prescribed: 1.3× even at catastrophic amplitude**
- correlated_schedule: 0.000050, prescribed: 0.000037
- The predictor compensates for a constant displacement: differences between timesteps are preserved
- 3 seeds, 200 episodes, 30 epochs, synthetic
- E29

**Ф43i. The free encoder (222×) is 167× worse than correlated noise (1.3×) at matched amplitude**
- free: 0.008282, correlated_schedule: 0.000050
- Drift ≠ a global coordinate shift. It is a data-dependent deformation.
- 3 seeds, 200 episodes, 30 epochs, synthetic
- E29

**Ф44i. Degradation spectrum: prescribed (1×) < correlated (1.3×) < noise_mid (6.2×) < FREE (222×) < noise_early (851×)**
- The free encoder sits closer to i.i.d. noise than to a correlated shift
- Drift introduces inconsistency *within* the context window, not merely a global offset
- 3 seeds, 200 episodes, 30 epochs, synthetic
- E29

### Critical window / sub-epoch (drift-hallucination branch, 03.07.2026)

**Ф45. The critical window for damage to the free encoder lies inside the first epoch (E30)**
- freeze@0 (random_fixed as proxy, Ф12 = 0.000476) → freeze@1 (0.06485, gym-pusht) = **136× cliff**
- freeze@1 → unfrozen (0.08113) = **1.3×** (negligible against the cliff)
- ~99% of the damage to the free encoder is done in the first epoch; the break is between epoch 0 and 1, not later
- The freeze@k≥1 band (k=1,2,3,5,7,10) is narrow: 0.065–0.082, each ≥25× worse than prescribed (0.00252)
- Mechanistically closes Ф31: random_fixed ≈ prescribed because both are freeze@0 (the representation BEFORE the catastrophic first epoch)
- Method: analysis of the already recorded freeze@k profile from E06/E07 all_results.json, no new runs; reproduces exactly (repro 136.25× = shipped)
- Environment: Push-T (gym-pusht, real pymunk, synthetic=false), 3 seeds, 30 epochs, 200 episodes
- Caveats: freeze@0 is a proxy (Ф12), not a literal run (normalization may differ); 3 seeds are a signal, not proof; there is no sub-epoch resolution; freeze is data-dependent (Ф30)
- Artefact: E30_critical_window/ (README + code + results, reproducible from all_results.json)
- E30

**Ф46. Within the first epoch the damage is a continuous SLOPE, not a discrete threshold (E31 synthetic → E32 real, SOLID)**
- Question: inside epoch 1, does the damage appear at a sharp threshold (a discrete point of no return) or accumulate as a slope? The encoder is frozen at fractions of the batches of epoch 1.
- **E31 (synthetic, 5 seeds):** verdict SLOPE (from analyze_shape.py). Linear-vs-step over the rise f≥0.25: the line is 2.2× better (SS 0.085 vs 0.186), linear R²=0.88. Monotone in 3/5 seeds (0 dips), 2/5 with a single noise dip.
- **E32 (real gym-pusht, 5 seeds {7,42,123,777,2024}):** verdict SLOPE, CLEANER than synthetic. **5/5 seeds strictly monotone** (0 dips of any kind). Pooled linear-vs-step over the band f∈[0.25,0.60] (per-seed min-max normalized): **linear R²=0.977**; the best single-breakpoint step is **10.5×** worse (SS_step 1.047 vs SS_lin 0.100, breakpoint f=0.50). Raw per-seed curves are slightly convex (increments grow toward f=0.60). E30-style anchor on real data: freeze@1.0/@0.0 = **22.1×** (per seed 7.2–35.3×) — the same direction as E30's 136× cliff.
- The first quarter of the epoch (f≤0.25) is near-harmless; after that the damage integrates continuously, accelerating toward the end
- Against a discrete irreversibility event; in favour of continuous drift. E32 removes the single (synthetic) caveat from Ф46 → **SOLID**.
- Environment: E31 — Push-T synthetic (synth(), 20 epoch/100 ep); E32 — Push-T real gym-pusht (reduced budget EP=4/NEP=50, pymunk 6.2.1 pinned)
- Caveats (E32): the reduced budget compresses the ABSOLUTE gaps (prescribed/unfrozen ~4× here vs 222× at scale; anchor cliff 22× vs E30's 136×) — do NOT compare these magnitudes with the 30-epoch runs; **the shape verdict is budget-robust** (the curve is monotone and slope-shaped regardless of budget). pymunk 6.2.1 (gym_pusht asks for ≥6.6, but 6.6+/7.x break add_collision_handler). A full-fidelity rerun (EP=15, NEP=200) is a one-line change.
- [NOTE for cross-checking] These artefacts contain NO resolution in (0.0, 0.25) (grid {0.0, 0.25…0.60, 1.0}); the near-harmless onset here is qualitative (f≤0.25). The claim "band [0.00–0.20] slope≈1.25 vs [0.25–0.60] slope≈13.04, ratio 10.4×" requires a separate higher-resolution run (sub-0.25) and is not supported by these files. The numbers above come from the shipped shape_verdict.json.
- Artefacts: E31_subepoch_freeze/, E32_subepoch_freeze_real/ (verdict in shape_verdict.txt / shape_verdict.json)
- E31, E32

---
---

### Environment: LLM activations (residual stream, last token position)

Internal-layer activations of decoder language models (transformer residual stream) at the last token position of the input prompt. Environment: decoder LLMs trained on next-token prediction; prompts as stimuli; activations are fixed (not fine-tuned). Metrics: PCA, LDA, residualization against one-hot auxiliary features, cosine cohesion, bootstrap stability.

**Our runs:** 5 LLMs (Qwen2.5-3B, Gemma2-2B, OLMo-1B, Falcon-1B, Pythia-1.4B); 80 prompts × 8 categories (factual, logical, spatial, emotional, abstract, code, ethical, narrative); PCA(40) on the residual stream at the last token position; residualization by linear regression on the one-hot last token (41 unique tokens); LDA with 5-fold stratified CV; bootstrap of 30 resamples at 70% (seed=42). The activations come from yadro_phase2 (Step 0, 12 April 2026).

**Ф47. R²(last token) on the top-7 PCs falls monotonically with depth in all 5 LLMs**
- Layer 0 → final layer: Δ = −0.24 to −0.32
- All 5 models: 4/4 steps down for Qwen/Gemma/OLMo/Falcon, 3/4 for Pythia
- At layer 0, R²=1.000 everywhere (identity: the last token's embedding *is* the last token)
- At the final layer R²=0.68–0.76 — the last token remains the dominant feature in the first 7 PCs
- E33 (Path 2)

**Ф48. LDA on 15 PCs after residualization by last token rises monotonically with depth in all 5 LLMs**
- Layer 0 → final layer: Δ = +0.18 to +0.25
- Final layer: 0.40–0.48; middle layer: 0.34–0.40
- Full monotonicity (4/4) for OLMo and Falcon; partial for Qwen, Gemma, Pythia
- E33 (Path 2)

**Ф49. Step 1 PCA was computed on middle layers in all 5 models**
- Qwen 18/36, Gemma 13/26, OLMo 8/16, Falcon 11/22, Pythia 16/32
- This is a consistent choice under the Step 0 protocol, not a Qwen-specific quirk
- The middle layer gives LDA on residuals of 0.34–0.40, the final layer 0.42–0.48
- Step 1 was carried out in a suboptimal layer for interpreting semantics
- E33 (Path 2)

**Ф50. The location of the semantic-signal peak on residuals differs between LLMs**
- Final layer: Qwen, OLMo, Falcon
- Mid-layer + plateau: Gemma (from layer 13)
- ≈0.75 of depth: Pythia (layer 24)
- This is a between-model difference and holds under any prompt design
- E33 (Path 2)

**Ф51. Paired duality of PC poles on residuals = 0 in all 5 LLMs**
- PAIRED criterion: both sides of the pole are homogeneous (normalized category entropy H<0.3)
- On raw PCA: 1–4 PAIRED PCs across models
- On residuals after residualization: 0 in all 5
- "Dual axes" are an artefact of the last-token confound
- E33 (Slice 1)

**Ф52. Asymmetric pole selection on residuals exists, but is distributed unevenly across categories**
- ASYMMETRIC PCs (one pole homogeneous): 2–7 across models
- Summed over 5 models: code 8, emotional 7, abstract 4, factual 3, logical 2, spatial 1, narrative 0, ethical 0
- E33 (Slice 1b)

**Ф53. Within-category cosine cohesion on final-layer residuals ranks the categories**
- Averaged over 5 models: emotional 0.128, spatial 0.086, code 0.086, abstract 0.083, factual 0.015, logical 0.003, narrative 0.000, ethical −0.019
- Cohesion is not "category quality" — it is a structural property of the activations
- E33 (Slice 1c)

**Ф54. Cohesion is positively associated with the frequency of asymmetric selection; the significance is not robust**
- Spearman ρ = 0.826, Pearson r = 0.746, N = 8 under exact decomposition (`svd_solver='full'`)
- Permutation null (200 label shuffles at fixed geometry, 163 valid, seed 42): mean −0.056, sd 0.397, q95 = 0.577, max = 0.764. No permutation reached the observed value; p_perm = 0.006 under the conservative (1+k)/(1+n) estimate. The association is not produced by a random partition
- The direction is stable: leave-one-model-out over the five models gives ρ 0.675–0.819, with no sign change on any subset
- The parametric p from `spearmanr` is not claimed at N = 8: across LOO it wanders over 0.013–0.066 and crosses 0.05 in two cases out of five
- The rank of spatial relative to code is decided by a cohesion gap of 1.267e−4 against a between-model SEM of 0.012 (code) and 0.018 (spatial), i.e. roughly 120× below the noise. This is a tie, not an outlier: swapping the two categories gives ρ = 0.7066. Dropping spatial takes N down to 7 and raises the significance threshold
- Consistent with Ф55
- E33 (Slices 1d, 1e), commit `a4bf9a7`

**Ф55. Categories as "nodes" at PC poles are not bootstrap-stable in 4 of 5 LLMs**
- 30 resamples of 70% of prompts (`N_BOOT=30`), PCA on 25 components (`N_PCA_COMP=25`), top-15 PCs after PCA+residualization, threshold ≥4/5 of one category at a pole
- Code is stable (≥70% of bootstraps) only in Falcon
- Emotional is stable in no model
- Categorical "nodes" in the current dataset are sampling effects, not a structural property of the activations
- E33 (Slice 2)

---

**Ф56. The E34 free encoder linearly encodes wall_x (R2 0.969) and barely encodes door_y (R2 0.211)**
- Ridge probe with cross-validated alpha from the frozen latent (512d, `final_ln` output) of the E34 free checkpoint, seed 1, epoch 11
- 4000 train / 1000 test episodes, freshly drawn from the E34 data config, held out by construction
- wall_x R2 = 0.9691, permutation floor +0.0011 +/- 0.0090; door_y R2 = 0.2109, floor -0.0038 +/- 0.0034
- Stable across frames: t=8 gives 0.9687 and 0.2138. Ridge alpha settled at 0.215 for both, so the fit does not lean on regularization
- Thresholds 0.7 and 0.3 were fixed in advance (question 8). wall_x clears 0.7; door_y falls below 0.3 while staying far above its floor, so it is weak rather than absent
- Structural control on the prescribed_2 encoder, an MLP over two numbers: wall_x 0.0388, door_y 0.0336, against 0.0091 and 0.0147 from its own input. The latent yields no more than it was fed, so the measurement does not manufacture the free result
- **Single checkpoint. Whether this holds across training seeds is not measured.** The claim is about this trained network, not about free encoders in general
- A linear probe measures linear decodability. door_y at 0.211 means it does not read out linearly, not that the door is absent from the latent
- Episodes could not be matched to the ones E34 trained on: `init_data` runs before `setup_seed` (`run_experiment_v3_windows.py:370-373`), so that draw was never seeded
- E34 (B1), `results/b1_output.txt`, `results/b1_control_output.txt`

---

### Environment: EB-JEPA Two Rooms (OBSERVATIONS, NOT FACTS)

The EB-JEPA Two Rooms environment (Meta FAIR, 2602.03604): goal-conditioned navigation in a two-room environment separated by a vertical wall with a door. wall_x and door_y are randomized between trajectories (fix_wall=False in the default config). Planning is done via MPPI in latent space.

**The E34 runs are single-seed (seed=1), 12 epochs, with nothing independently repeated. Under the programme protocol these are NOT FACTS but observations. They are recorded as Н1–Н4 to register the signal, not as load-bearing results.**

**Н1. Single-seed prescribed_2 (only x_a, y_a) gives SR=0%, free (pixels) gives SR=55% on Two Rooms**
- 12 epochs, batch=64, seed=1, 100K episodes
- Prescribed encoder: MLP 2→256→256→512, 199K params
- Free encoder: ImpalaEncoder, 1.43M params
- Probe loss is **not comparable across conditions**: in the prescribed condition the probe target is the encoder's own input (`run_experiment_v3_windows.py:490-494`), while in the free condition it must be extracted from pixels. The 12× figure was withdrawn once the probe target was checked
- Pred loss for free (0.024) beats prescribed (0.051) by 2.2× — the predictor learned the dynamics in both cases
- **The conditions fall short of a fact under the programme protocol:**
  - 1 seed (minimum 3)
  - 12 epochs (LeCun et al. report ~97% SR — possibly at larger epoch counts or a different configuration)
  - Variance was not estimated
- Alternative explanations for the 0% of prescribed:
  (a) insufficient coordinate dimensionality (tested in E35 prescribed_4)
  (b) insufficient training length
  (c) architecture mismatch (MLP vs CNN on the same predictor)
  (d) sim_loss is specifically harmful to prescribed (see Н3)
- E34

**Н2. The distance distribution of prescribed_2 episodes in planning eval is bimodal**
- 4/20 episodes with dist<20 (close to the goal)
- 16/20 episodes with dist>20 (far from the goal)
- Hypothesis: successes occur when start and goal are in the same room, failures when the door must be used
- **Requires verification against the start/goal coordinates of each episode** — without that check this is a distribution, not evidence for the hypothesis
- This hypothesis is the motivation for E35 prescribed_4 (giving the encoder explicit information about the wall and the door)
- E34

**Н3. sim_loss for prescribed_2 rises across epochs while pred_loss falls**
- sim_loss: epoch 0 = 0.010, epoch 11 = 0.017
- pred_loss: epoch 0 = 0.174, epoch 11 = 0.051
- A structural analogue of Ф7/Ф8 (SIGReg is harmful to anisotropic prescribed on Push-T)
- A candidate for a `prescribed_no_sim` ablation — but on a single seed this cannot count as a fact
- The free encoder shows the opposite: sim_loss falls (0.016→0.008) together with pred_loss
- E34

**Н4. Mean-of-distribution shift — free shifts more relative to its norm than prescribed**
- Relative shift |Δμ(0,t)|/|μ_t|: free 0.69–0.78, prescribed 0.50–0.63
- By ep11: cosine(μ_0, μ_11) — prescribed 0.91, free 0.69
- Step-by-step pairwise consistency (avg cos): prescribed −0.02, free −0.05 (both incoherent, "wandering")
- **Caveat:** these are aggregates over 20 of 512 dims after the encoder's LayerNorm; **not comparable** with the per-sample R² transfer drift from E18
- The direction agrees with the Paper 2 thesis (free drifts more), but it is not a strict analogue
- E34 (post-hoc analysis of encoder_stats.json, 30.04.2026)

---

## HYPOTHESES

### Confirmed (on a single environment)

**Г1. Fixing the axes matters more than the semantics of the axes**
- Confirmed: random fixed ≈ prescribed (Ф5, Ф12, Ф31)
- Strengthened: random_fixed_3d ≈ prescribed ≈ rotated_prescribed (Ф31) — alignment within the subspace does not matter
- Reproduced in 5D (Ф34: random ≈ prescribed, 0.92×)
- Begins to weaken in 16D (Ф34: 1.53×) — with non-linear features, alignment starts to appear
- Environment: Push-T (3D, 5D)
- Status: CONFIRMED on Push-T, with a caveat for high dimensions

**Г2. Coordinate drift is the cause of free-encoder degradation**
- Confirmed: R² < −62 (Ф10), freeze@1 +20% (Ф11), aligned-but-drifting ≈ free (Ф15)
- Strengthened (Tier 1):
  - Early drift destroys information — the MLP decoder breaks too (Ф24)
  - Late drift preserves information non-linearly (Ф25)
  - The drift is non-linear — PCA cannot fix it (Ф27)
  - diffLR 100× still leaves a 62× gap — not optimization lag (Ф26)
- No confound in the freeze test (Ф30)
- Drift grows with dimension (Ф35)
- Environment: Push-T (3D, 5D, 16D)
- Status: CONFIRMED, strengthened

**Г3. Rank collapse is not the main cause of free-encoder degradation**
- Confirmed: full rank 2.99 + isotropy 0.86 → still 233× worse (Ф9)
- Environment: Push-T
- Status: CONFIRMED on Push-T

**Г4. Stability is the prerequisite, alignment an additional factor**
- Confirmed: 2×2 factorial (Ф16), aligned-but-drifting ≈ free (Ф15)
- Refined (Tier 2): alignment *within the subspace* does not matter (Ф31)
- Alignment = choosing the right subspace + normalization, not the orientation of the axes
- Environment: Push-T
- Status: CONFIRMED, refined — "alignment" redefined as "subspace selection"

**Г19. The LLM residual stream at the last token position moves away from the token identity and toward a semantic representation with depth**
- R²(token) on the top-7 PCs falls monotonically with depth in all 5 LLMs (Ф47)
- Category LDA on 15 PCs after residualization rises monotonically (Ф48)
- At layer 0 PCA = token identity (R²=1.000); at the final layer the token is weakened and semantics strengthened
- Consequence: final layers are preferable to middle layers for semantic interpretation
- Environment: 5 LLMs (Qwen2.5-3B, Gemma2-2B, OLMo-1B, Falcon-1B, Pythia-1.4B)
- Status: CONFIRMED on a single environment (E33)

**Г20. The apparent categorical structure in the leading PCs of the residual stream is an artefact of the last-token effect plus sampling compactness of the categories**
- There are no paired dual poles on the residuals (Ф51): 1–4 PAIRED PCs on raw PCA, 0 on residuals in all 5 models
- Asymmetric nodes do exist (Ф52), and their frequency is positively associated with within-category cohesion (Ф53, Ф54): Spearman ρ≈0.83, significance not robust at N = 8, see Ф54
- But the nodes are not bootstrap-stable in 4 of 5 models (Ф55) — they depend on the prompt sample
- Consequence: Step 1 interpretations of the "geometry X-centric" / "factorized geometry" kind are not supported by the data once the last-token confound is controlled
- Environment: 5 LLMs, the current yadro_phase2 dataset (80 prompts, heterogeneous syntactic endings)
- Status: CONFIRMED on a single environment, with the caveat that the result may depend in part on the choice of prompts (E33)

**Г21. The location of the semantic-signal peak in the residual stream is a between-model difference, not a methodological artefact**
- Final layer for Qwen, OLMo, Falcon; mid+plateau for Gemma; ≈0.75 for Pythia (Ф50)
- Holds after residualization by last token, on the same prompts
- Environment: 5 LLMs
- Status: CONFIRMED on a single environment; needs further checking on more models and on controlled prompts (E33)

**Г18. The critical window for free-encoder damage is inside the first epoch; the shape is a continuous slope, not a discrete threshold**
- Refines Г15 (the two-phase model): phase 1 is not "epochs 0–2" but "inside the first epoch".
- Support: Ф45 (E30) — ~99% of the damage in the first epoch (freeze@0→@1 = 136× cliff vs freeze@1→unfrozen = 1.3×); Ф46 (E31 synthetic + E32 real) — inside the first epoch it is a slope (SLOPE verdict from code: E32 R²=0.977, step 10.5× worse, 5/5 monotone).
- The first quarter of the epoch (f≤0.25) is near-harmless; after that the damage integrates continuously, accelerating toward the end.
- **The correct formulation (important for the hallucination bridge):** "early, continuously-integrated divergence that later training does not undo" — NOT "irreversible event". The word "irreversible" must not be read as discreteness: irreversibility is a property of the terminal state of epoch 1 with respect to later training (E30/Ф45), reached by continuous accumulation (E31/E32/Ф46). The slope directly refutes a discrete threshold.
- Bridge to hallucination: the shared axis is confident output from an ungrounded state (NOT an identity of mechanism). The real domain test is the LLM (Г17, epiplexity ⊥ identifiability), outside Push-T. Keep it on the leash.
- Environment: Push-T (synthetic E31 + real gym-pusht E32). The hallucination domain is separate (Г17).
- Status: CONFIRMED on Push-T (E30 real + E31 synthetic + E32 real). E32 caveat: reduced budget → absolute magnitudes compressed, shape verdict budget-robust.

---


### Refuted

**Г5. The prescribed/free crossover equals the intrinsic dimension of the task**
- Push-T: ~~crossover 3→4~~ NO CROSSOVER on full data (E28). Prescribed wins at 1–11.
- Pendulum: no crossover, intrinsic dim=2 — (data from Ф19, not verified with full parameters)
- Double pendulum: crossover 1→2, intrinsic dim=4 — (data from Ф20, being rerun)
- The original Push-T crossover was an artefact of the underpowered E13 (100 ep, 20 epochs, 2 seeds)
- Status: REFUTED (no crossover exists on Push-T at full parameters)

**Г6. 11 axes (by analogy with M-theory) will give a better result**
- prescribed_11 is 20× worse than prescribed_3 (Ф17)
- Status: REFUTED

**Г10. The prescribed advantage is a pure stability effect (17× × 13× = 233× decomposition)**
- The Paper 2 decomposition is invalid:
  - random_fixed_5d (from Ф12) is unstable — it blows up on new seeds (Ф32)
  - random_fixed_3d (from the correct subspace) ≈ prescribed (Ф31)
  - The "17× stability" was an artefact of one particular random_fixed implementation
- The correct decomposition: subspace selection + normalization + freeze
- Alignment of the axes within the subspace is not a factor
- Status: REFUTED (the decomposition, not the thesis)

**Г11. The free-encoder problem is optimization lag (solved by scheduler/LR)**
- diffLR 100× helps by 72%, but a 62× gap remains (Ф26)
- Extra predictor steps (K=3) make it WORSE (Ф26)
- EMA (decay=0.996) does not help (Paper 2, 6.1×)
- Status: REFUTED — optimization lag is a partial factor, not the main cause

**Г12. Drift is rotation/scaling, solved by PCA canonicalization**
- PCA degrades R² transfer at most epochs (Ф27)
- The drift is non-linear
- Status: REFUTED

**Г13. SIGReg destroys the aligned initialization**
- SIGReg stabilizes aligned-drifting linear and prevents divergence (Ф29)
- Without SIGReg: seed 123 → catastrophic divergence (1.39)
- Status: REFUTED — SIGReg stabilizes, it does not destroy

**Г22. Step 1 PCA on yadro_phase2 shows clean semantic geometry of LLM categories**
- On raw PCA of the final and middle layers, the leading PCs are 68–86% explained by the last token (Ф47)
- Category LDA after residualization falls by ~2× on 7 PCs and by ~1.5–2× on 15 PCs
- The PC parameters on which the Step 1 interpretations were built are dominated by a first-order effect, not by semantics
- Status: REFUTED — Step 1 PCA in its current form shows a mixed signal; the clean semantics on residuals is weaker than the first-order artefact (E33, all slices)

**Г23. Categories form dual pairs at the poles of LLM residual-stream PCs**
- On residuals after residualization, PAIRED PCs = 0 in all 5 models (Ф51)
- On raw PCA there are 1–4 PAIRED PCs across models — but this is explained by the group structure of the categories' last tokens, not by semantics
- The specific pairs previously conjectured ("ethical↔spatial", "logical↔emotional") do not reproduce on residuals (Ф51, Ф52)
- Status: REFUTED — the pairing disappears once the last-token confound is controlled (E33, Slice 1)

**Г24. Categorical nodes at PC poles are a structural property of LLM activations**
- On the full sample the nodes exist (asymmetric, Ф52)
- Bootstrap check (30 resamples at 70%, ≥70% threshold): code is stable only in Falcon (1/5 models); every other category and model is unstable (Ф55)
- The nodes depend on the prompt sample and do not reflect a structural property of the activations
- Status: REFUTED on the current dataset — testing this at the structural level requires controlled prompts with more examples per category (E33, Slice 2)

**Г16. The free latent has a characterizable drift rate (drift-rate law)**
- Imported from INS/GNSS-denied work (SBG Systems, triage 07.2026): dead reckoning models drift as error growth — accumulated error = drift rate × time since the anchor was removed. Transfer: the accumulated deviation of the free encoder's basis grows as a function of the number of epochs without a prescribed constraint.
- Partly observed already (raw material for a rate fit exists, 3 seeds):
  - Ф10: per-epoch R² transfer (seed 42 −16.9, 123 −62.2, 777 −25.4; recovery by ep 2→3)
  - Г15: the two-phase model, drift rate ≈ 0.3 as the phase boundary (from the T8 scatter)
- New prediction (testable on Push-T): the accumulated basis deviation (Procrustes to the anchor truth) is a monotone function of epochs-since-anchor-removal with an estimable rate; drift-rate(free) >> drift-rate(prescribed ≈ random_fixed ≈ 0).
- Measurement method (imported from SBG):
  - ablation-by-withholding: the anchor is withheld from the encoder but kept as ground truth for measurement (analogous to the Qinertia GNSS rejection module — simulating an outage without physically switching anything off)
  - measure the trajectory of the basis across epochs, NOT the endpoint: an endpoint metric (e.g. final Procrustes R²=−15.7) masks the drift of the process — a direct analogue of the "loops and turns mask heading drift" confound and of the E33 last-token confound
- Horizon conjecture (NOT tested on Push-T, cross-domain — not a fact): exceeding a threshold of accumulated drift = transition into a hallucination basin (2604.04743, static basins). drift-rate × t → threshold would give the dynamics of the transition. A separate drift→hallucination paper.
- Environment: Push-T (a rate fit is possible from the existing E-data of Ф10/Г15). The horizon part is out of environment.
- **REFUTATION (03.07.2026, checked against all_results.json, no new runs):**
  - [FACT] free raw_drift falls 1.43 (ep1) → 0.001 (ep30); late/early ratio = 0.002 — the drift burns out in ~3 epochs.
  - [FACT] Cumulative fit: linear (constant rate) R²=0.821 < sqrt (diffusion) 0.920 < log (saturation) 0.977. There is NO constant rate — the drift is a decaying front-loaded transient, not an accumulation.
  - [INFERENCE] The INS/GNSS dead-reckoning analogy is false: in an INS the error grows without bound over time, here it self-extinguishes. There is nothing to build "drift-rate × t → threshold" on.
  - [INFERENCE] The data support Ф24/Ф25 (the two-phase model), NOT Г16.
  - Caveat: "log R²=0.977" ≠ "a logarithmic law" — it is an artefact of front-loading; prescribed=0 is degenerate.
  - Artefact: E30_critical_window/finding_drift_rate.docx
- Status: **REFUTED** — the rate form (a constant/characterizable rate) is not supported by the data; the drift is a front-loaded decaying transient (phase 1 of Ф24/Ф25). The drift→hallucination horizon conjecture loses its kinematic support in this form.


### Open

**Г7. ~~Prescribed works not because of fixing but because of information selection~~ REFUTED**
- Push-T: prescribed wins at dim=5 (all coordinates, no selection) by 66× (E28)
- Double pendulum: prescribed_norm wins at dim=4 (all coordinates, no selection) by 19× (Ф20)
- The original argument (pendulums do not work → selection is needed) was an artefact of missing normalization
- Status: REFUTED — fixing + normalization suffices, selection is not needed

**Г8. ~~Prescribed works because of the combination of fixing + selection~~ REFUTED**
- Fixing without selection works on both environments given normalization:
  - Push-T dim=5 (all coordinates): 66× (E28)
  - Double pendulum dim=4 (all coordinates): 19× (Ф20 updated)
- Selection is not a necessary component
- Correct formulation: prescribed = fixing + [0,1] normalization
- Status: REFUTED

**Г9. Fragility of prescribed: one extra axis kills the advantage**
- ~~Fact: dim=3→4 is a loss on Push-T (Ф18)~~ REFUTED: prescribed wins at dim=4 by 114× (E28)
- The mechanism from E17 (Ф21–Ф23) still holds: an unpredictable axis is catastrophic (1106×)
- But: predictable extra axes degrade prescribed (228× → 42× over dim 3→11) without killing it
- The gap shrinks monotonically, but prescribed wins at every dimension
- **E12 (free_11 > prescribed_11) was an artefact of a small predictor (hidden=128)**
- With predictor capacity max(128, dim*8) prescribed wins even at dim=11
- Status: PARTIALLY REFUTED — fragility to noise axes is real (Ф21), but extra predictable axes do not kill the prescribed advantage

**Г14. The prescribed advantage = fixing + [0,1] normalization of the coordinates**
- Based on all the experiments (Tier 1–3, E28, the E16 fix)
- Prescribed wins when it:
  (a) fixes the coordinates (they do not drift)
  (b) **normalizes the coordinates into [0,1] (min-max)** — MANDATORY
  (c) has coordinates that carry information relevant to the task
- Without normalization prescribed loses (pendulum raw: 0.1–0.4×)
- With normalization prescribed wins on every environment and every dim:
  - Push-T dim 1–11: 42–1820× (E28)
  - Double pendulum dim 1–8: 10–19× (Ф20 updated)
- Alignment of the axes within the subspace does not matter (Ф31)
- random_3d ≈ prescribed under normalization (Ф31, Ф34)
- Standardization (zero-mean, unit-var) degrades by 15× (Ф40)
- **Given enough data (500 ep) the free encoder beats prescribed by 695,000× (Ф38)**
- Prescribed = sample efficiency + guaranteed stability, not an absolute advantage
- Gauge fixing does not work as an alternative (Ф37)
- Status: CONFIRMED on Push-T and the double pendulum (2 environments, 3+ dimensions)

**Г15. Two-phase model of drift**
- A new hypothesis based on Tier 1
- Phase 1 (ep 0–2): catastrophic — information is destroyed non-linearly and unrecoverably
- Phase 2 (ep 3+): stabilization — information is preserved, but in a non-linearly drifting space
- Phase boundary: drift rate ≈ 0.3 (from the T8 scatter plot)
- The R² ceiling of the free encoder (phase 2) is ≈ 0.75 (linear); the gap to prescribed is set by alignment, not drift
- Status: NEW, confirmed on Push-T 3D (Ф24, Ф25, Ф28)

**Г17. Epiplexity ⊥ identifiability (structural information is blind to coordinate drift)**
- Thesis: epiplexity (S_T, structural information in the sense of time-bounded MDL, Finzi et al. arXiv:2601.03220) measures the *amount* of absorbed structure while remaining agnostic to its groundedness/identifiability. A model can absorb a large amount of structure (low NLL → high S_T) with broken coordinate identifiability (R² < 0 after Procrustes). → epiplexity and coordinate stability are orthogonal axes.
- Support (their own stated properties):
  - epiplexity is content-agnostic by definition (Finzi §6.1): it measures the amount of structural information irrespective of its content.
  - epiplexity is loss-derived (prequential = the area under the loss curve above the final value). It sees the volume of compression, not the geometry of the basis. → the strongest instantiation of the thesis "space matters more than the loss": even an information-theoretic reading of the loss curve fails to distinguish grounded structure from an internally consistent drifting code.
- What is observed so far is only low-id: free R²=−15.7 (Procrustes, final) with full rank/isotropy (Ф9) and a decent NLL. This is an Н (1 observation).
- What is NOT observed (yet): the epiplexity of free/prescribed encoders has not been measured. The "high-epi + low-id" cell is a PREDICTION, not a fact.
- Confound (a potential contributor): the prequential estimator assumes convergence to a stable final loss. Under drift the "final" value is ill-defined → S_preq may be inflated by non-convergence, conflating "absorbed structure" with "inability to ground". Finzi's paper does not check this assumption. A direct analogue of the endpoint-vs-trajectory confound (Г16) and of the last-token confound (E33).
- Domains (do not conflate): in JEPA, epiplexity is only a proxy (there is no explicit likelihood) — do NOT drag it in there as a metric. Its proper home is the LLM latent (a likelihood model), i.e. the drift→hallucination thread, not the prescribed-axes papers.
- Difference from Г16 (stated explicitly so the two do not merge): Г16 = drift is characterizable as a rate in time (kinematics, dead reckoning). Г17 = a metric of the amount of structure does not see drift at all. Different claims; only the hallucination horizon is shared. Do not merge.
- Test (likelihood domain): on LLM checkpoints, S_preq (from existing NLL logs) × a drift/identifiability metric for the latent. Confirmation = the high-epi/low-id cell is non-empty and S_preq fails to distinguish grounded from drifting structure when split by identifiability. Refutation = S_preq correlates monotonically with identifiability (in which case epiplexity is a symptom, not an orthogonal axis).
- Environment: outside Push-T (the domain is the LLM). Not applicable as a metric for JEPA.
- Status: OPEN / HYPOTHESIS. Support = 1 Н (low-id) + an untested prediction (epiplexity not measured). NOT a fact: an LLM test on ≥3 seeds with explicit measurement of S_preq is required. Links: Paper 1 "Space Matters More Than the Loss"; Г16; hallucination basins (2604.04743); a separate drift→hallucination paper after phase 2 of Yadro.

**Г25. The prescribed advantage requires coordinate completeness with respect to the downstream task**
- Identifiability of the axes does not substitute for completeness of the coordinate description
- Single-seed observation on Two Rooms: prescribed_2 = (x_a, y_a) gives 0% planning SR against 55% for free (Н1). Agent coordinates without information about obstacles do not let MPPI plan through the door — in that latent space the door does not exist
- Falsifier: if prescribed_4 = (x_a, y_a, wall_x, door_y) gives SR ≈ 0% on Two Rooms, the completeness hypothesis is refuted and the problem is deeper (architecture mismatch, insufficient capacity, a fundamental limitation of the prescribed approach in environments with obstacles)
- Test: E35 prescribed_4 (planned, expected compute 60–100 h CPU)
- Status: OPEN, goes to test in E35

---

## KEY DIFFERENCES BETWEEN ENVIRONMENTS

| Property | Push-T | Double pendulum (raw) | Double pendulum (norm) | EB-JEPA Two Rooms (Н1) |
|---|---|---|---|---|
| Intrinsic dimension | 3 | 4 | 4 | 2+ (agent x, y; wall, door are environment parameters) |
| Full state dim | 5 | 4 | 4 | 65×65 pixels + (wall_x, door_y) |
| Prescribed wins? | Yes (dim 1–11) | No (Ф20 original) | **Yes (dim 1–8)** | No at dim=2 (Н1, single seed) |
| Normalization | [0,1] min-max | None | [0,1] min-max | z-score (LeCun) for prescribed_2 |
| Gap prescribed/free | 42–1820× | 0.1–0.4× | **10–19×** | Free ≫ prescribed (free 55%, prescribed 0% SR) |
| Obstacles in the environment | No | No | No | Yes (a wall with a door) |
| Downstream task | Prediction loss | Prediction loss | Prediction loss | Goal-conditioned planning (MPPI) |

Two Rooms is the first environment in the programme with (a) obstacles and (b) a downstream metric other than prediction loss. Both differences may affect the result. E35 prescribed_4 tests whether coordinate completeness helps in this environment.

---

## KEY CONTRADICTIONS (require resolution)

**П1. ~~Push-T 5D prescribed works (Ф36), but pendulum prescribed on the full input does not (Ф19, Ф20)~~ CLOSED**
- Cause: the absence of normalization in the pendulum runs
- With [0,1] normalization prescribed beats free on the double pendulum at ALL dims 1–8 (Ф20 updated)
- Normalization gives a 37–166× improvement to prescribed
- Push-T prescribed always normalized — which is why it worked
- The pendulums without normalization use raw coordinates at different scales (θ ∈ [−π,π], ω ∈ [−10,10])
- Ф19 (simple pendulum) is presumably the same artefact; verification needed

**П2. ~~Dim sweep (Ф18): prescribed loses at dim≥4. But Ф36: prescribed_5d wins by 66×~~ CLOSED**
- E13 (100 ep, 20 epochs, 2 seeds) was underpowered
- E28 (200 ep, 30 epochs, 3 seeds, predictor max(128, dim*8)): prescribed wins 1–11, no crossover
- dim=5 agrees with Tier 3 E25: 66.3× vs 66.2×
- Cause of the discrepancy: (a) too little data/epochs, (b) a small predictor at hidden=128
- Ф18 refuted, Ф36 confirmed

---

## OPEN QUESTIONS (by priority)

1. ~~Is selection (feature selection) rather than fixing the true cause of the prescribed advantage on Push-T?~~ → Partly answered: Push-T 5D prescribed (without selection) works (Ф36). Selection is not the only factor. But the pendulums without selection do not work — the difference needs explaining (П1)
2. Why does Push-T 5D prescribed work while pendulum prescribed on the full input does not? (П1) → E16 is being rerun with normalization
3. ~~Why do the dim sweep (Ф18) and Tier 3 (Ф36) give different results at dim=5?~~ → SOLVED (П2): E13 was underpowered. E28 confirmed that prescribed wins at 1–11.
4. ~~Why does even sinθ (same subspace) degrade things by 4.8×?~~ → The context has changed: prescribed wins at every dim; the degradation is a loss of gap magnitude, not a defeat
5. ~~Do the facts Ф17–Ф23 reproduce on full data?~~ → Ф17 updated (E28): prescribed_11 now beats free_11 (42×) given the right predictor capacity. Ф21–Ф23 (fragility) need a rerun with the max(128, dim*8) predictor.
6. How does prescribed behave in environments with dim_state > dim_internal, under normalization? → Partly: E16 is being rerun
7. Does the prescribed advantage carry over to environments with obstacles (where the downstream task is planning, not prediction loss)? → Single-seed E34 (prescribed_2) gives 0% SR. Being tested in E35 prescribed_4 = (x_a, y_a, wall_x, door_y). Related to Г22.
8. ~~What is in the latent space of the Two Rooms free encoder — did it implicitly learn wall_x/door_y from pixels?~~ → ANSWERED (Ф56, B1, local CPU run): wall_x yes (R² 0.969, above the 0.7 threshold), door_y almost not (R² 0.211, below 0.3 but above its floor). The split was not anticipated by the question, which assumed both coordinates would move together. It shifts the reading of Н1: the contrast between conditions is about the obstacle. How free reaches the opening while encoding its position so weakly is a new open question (see 9).
9. How does the free encoder reach the door while encoding door_y at only R² 0.211 linearly (Ф56)? Candidates: the door is represented nonlinearly (separable by an MLP probe on the same latents but not by ridge); it is represented locally rather than as a global coordinate, so a single frame near the agent carries it only when the agent is close; or planning does not use a door representation at all and MPPI finds the opening reactively. The first is cheap to test on the existing checkpoint, the third needs the planner.

---

## PROTOCOL

- Facts: only experimental results, with parameters
- Hypotheses: with an explicit status (confirmed / refuted / open)
- Observations (Н): single-seed or underpowered results, marked explicitly as observations rather than facts. They are recorded to register a signal, not as load-bearing support for hypotheses
- Confirmation on one environment ≠ confirmation in general — always state the environment
- When experiments contradict each other, record the contradiction (П) together with possible explanations
- Update with every new experiment

---

## MAPPING TABLES (merge of 20.08.2026)

The April and July branches of the registry developed in parallel and independently reused the numbers E30–E34 and Г16–Г22. The July numbers are committed in `648f1fd` and are referenced by the experiment READMEs, so it is the April branch that was renumbered.

### Experiments

| April number | New number | What it is |
|---|---|---|
| E30 (PLANNED) | **E36** | Full coordinate drift on vision SSL. Planned. |
| E31 | **E33** | Step 1 PCA: last-token confound and pole stability on 5 LLMs. Ф47–Ф55, Г19–Г24. |
| E32 | **E34** | EB-JEPA Two Rooms — prescribed_2 vs free planning. Н1–Н4 (single seed, NOT facts). |
| E33 | **E35** | EB-JEPA Two Rooms — prescribed_4 (wall and door coordinates). READY_TO_START, Г25. |
| E34 (DEFERRED) | **E37** | CARLA prescribed safety axes. Deferred. |

The July E30, E31, E32 (critical window, sub-epoch freeze synthetic, sub-epoch freeze real) keep their numbers. PreE30 (the DINOv2 pilot) has no collision. E38 onward are free.

### Hypotheses

| April number | New number | What it is |
|---|---|---|
| Г16 | **Г19** | The LLM residual stream moves away from token identity with depth. |
| Г17 | **Г20** | Categorical structure in the leading PCs is an artefact of the last-token effect. |
| Г18 | **Г21** | The location of the semantic-signal peak is a between-model difference. |
| Г19 | **Г22** | Step 1 PCA shows clean semantic geometry (refuted). |
| Г20 | **Г23** | Categories form dual pairs at PC poles (refuted). |
| Г21 | **Г24** | Categorical nodes are a structural property of the activations (refuted). |
| Г22 | **Г25** | The prescribed advantage requires coordinate completeness. |

The July Г16 (drift-rate law, refuted), Г17 (epiplexity ⊥ identifiability, open) and Г18 (critical window = slope, confirmed) keep their numbers.

### What else changed in the merge

- The July Г16 and Г18 physically sat in the "Open" section while carrying the statuses REFUTED and CONFIRMED. They were moved into the sections matching their status. Г17 stays in "Open".
- References to `last-token confound (E31)` inside the July Г16 and Г17 pointed at the April LLM work and were rewritten to E33. References to `E31 synthetic + E32 real` inside Ф46 and Г18 point at the July experiments and were left unchanged.
- Lost from the July `EVIDENCE.md` at the fork and restored from the April branch: the full protocol including the definition of the class **Н (observations)**; the EB-JEPA Two Rooms column in the KEY DIFFERENCES table together with the rows "Obstacles in the environment" and "Downstream task"; questions 7 and 8 in OPEN QUESTIONS.
