# Experiment registry: Prescribed Axes

Author: Andrey Lazarev | Started: March 2026
Last updated: 20 August 2026 — merge of the April and July branches of the registry.

---

## Numbering

- **E01–E05**: Paper 1 (prescribed-axes)
- **E06–E12**: Paper 2 (prescribed-axes-drift)
- **E13–E18**: Dim sweep
- **E19–E21**: Tier 1 critical tests
- **E22–E24**: Tier 2 confound tests
- **E25–E27**: Tier 3 generalization tests
- **E28–E29**: П2 resolution + drift nature controls
- **PreE30**: Pilot — coordinate drift on DINOv2 (production-scale vision SSL)
- **E30–E32**: Drift / hallucination branch (critical window, sub-epoch freeze synthetic + real)
- **E33**: Step 1 PCA diagnostic — last-token confound and pole stability (LLM activations). *Was E31 in the April branch.*
- **E34**: EB-JEPA Two Rooms prescribed_2 vs free — single-seed observation. *Was E32.*
- **E35**: EB-JEPA Two Rooms prescribed_4 — closing the E34 gap, testing Г25 (READY_TO_START). *Was E33.*
- **E36**: Full coordinate drift on vision SSL (PLANNED, see PreE30). *Was E30.*
- **E37**: CARLA prescribed safety axes (DEFERRED). *Was E34.*
- **E38+**: free. The nearest candidate is ECA / epiplexity (Г17).

> **Numbering collision (discovered 20.08.2026).** The April and July branches of the registry developed in parallel and independently used the numbers E30–E34 and Г16–Г22. The July numbers are committed in `648f1fd` and are referenced by the experiment READMEs and by Ф45/Ф46 — so it is the April branch that was renumbered. The mapping table is at the end of this file and in `EVIDENCE.md`.

---

## Paper 1: The Space Matters More Than the Loss

### E01. Speech JEPA: prescribed cluster anchors vs free
- **Environment:** LibriSpeech
- **Conditions:** 2×2 factorial {GMM, k-means} × {soft, hard} vs pure JEPA
- **Metric:** Cluster entropy (codebook utilization)
- **Result:** +18–20pp entropy for prescribed. Soft ≈ hard (Δ<0.03%). Frozen structure is the dominant factor.
- **Parameters:** Pilot
- **Facts:** Ф3
- **Code:** prescribed-axes repo
- **Data:** —

### E02. Shov-JEPA: 3 prescribed axes vs 64 free (Rico UI, vision)
- **Environment:** Rico dataset, 398 UI screenshots
- **Conditions:** ShovJEPA (3 axes: position, functionality, depth) vs free 64D
- **Metric:** Validation accuracy
- **Result:** 72.5% vs 67.5% (+5%)
- **Parameters:** 398 samples, single seed, pilot
- **Facts:** Ф4
- **Code:** prescribed-axes repo (shov-jepa)
- **Data:** shov-jepa-report-ru.docx

### E03. LeWM State: prescribed 3D vs free 3D (Push-T)
- **Environment:** Push-T (gym-pusht, pymunk physics)
- **Conditions:** Prescribed = normalize(x_b, y_b, θ_b) vs free MLP 5→3 + SIGReg
- **Metric:** Val prediction loss
- **Result:** Prescribed 0.004, free 0.157 = **38×**. Per axis: x 53×, y 63×, θ 25×
- **Parameters:** 3 seeds, 50 epochs, 200 episodes, SIGReg block
- **Facts:** Ф1
- **Code:** prescribed-axes repo (lewm_state)
- **Data:** lewm_state_results/

### E04. LeWM Pixel: prescribed 3D vs free CNN (Push-T from pixels)
- **Environment:** Push-T (96×96 pixel observations)
- **Conditions:** Prescribed 3D (20K params) vs free CNN (744K params)
- **Metric:** Val prediction loss
- **Result:** Prescribed **14.8×** better with **37× fewer parameters**. The CNN plateaus at epoch 7.
- **Parameters:** 50 epochs
- **Facts:** Ф2
- **Code:** prescribed-axes repo (lewm_pixels)
- **Data:** lewm_pixels_results/

### E05. Controls: random fixed, equal-input, SIGReg ablation (Push-T)
- **Environment:** Push-T
- **Conditions:** Random fixed 3D, free 3D same input, ±SIGReg
- **Metric:** Val prediction loss
- **Result:**
  - Random fixed ≈ prescribed (0.61×) → fixing > semantics (Ф5)
  - Equal-input free is 7.6× worse than prescribed → not about access to information (Ф6)
  - SIGReg removal improves free by 1.9× (Ф7)
- **Parameters:** 3 seeds, 50 epochs, 200 episodes
- **Facts:** Ф5, Ф6, Ф7
- **Code:** prescribed-axes repo (reviewer_response_experiments.py)
- **Data:** reviewer_results/summary.json, reviewer_results/full_results.json

### E05a. Random axes scaling: 200ep vs 500ep (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed, random_fixed, free_3d, free_5d at 200 and 500 episodes
- **Metric:** Val prediction loss
- **Result:**
  - 200 ep: random 0.61× prescribed, free 4.47× worse (Ф39)
  - 500 ep: random 1.00× prescribed, **free 695,000× BETTER** (Ф38)
  - Fixed encoders plateau at ~8.5×10⁻⁴, free → 10⁻⁹
  - Prescribed = sample efficiency, not absolute superiority
  - Isotropic normalization 15× worse (Ф40)
- **Parameters:** 200 ep: 3 seeds, 30 epochs. 500 ep: 3–9 runs, 50 epochs. No SIGReg.
- **Facts:** Ф38, Ф39, Ф40
- **Code:** random_axes_control/run_random_axes_control.py, run_isotropic_control.py
- **Data:** exp5_random_axes/all_results.json (18 runs), random_fixed_results/results.json

### E05b. Gauge fixing the free encoder (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed, free, gauge_fixed_free, linear_free
- **Metric:** Val prediction loss
- **Result:**
  - gauge_fixed_free 1.08× ≈ free — gauge fixing does not help (Ф37)
  - linear_free: 16009 (blow-up)
- **Parameters:** Data seed 42, training seeds [42, 123, 777], 50 epochs, synthetic
- **Facts:** Ф37
- **Code:** (gauge_fix experiment script)
- **Data:** gauge_fix_results/results.json

---

## Paper 2: Semantic Drift, Not Rank Collapse

### E06. Covariance + drift analysis (Push-T, gym-pusht)
- **Environment:** Push-T (gym-pusht, real pymunk physics)
- **Conditions:** Prescribed vs free, covariance at sampled epochs, drift metrics
- **Metric:** Effective rank, isotropy, raw/aligned drift, R² transfer
- **Result:**
  - Free: rank 2.99, isotropy 0.86 → loses to prescribed by 233× (Ф9)
  - R² transfer epoch 0→1: −16.9 / −62.2 / −25.4 (Ф10)
  - 80% of the drift is structural after Procrustes
  - SIGReg harms free: 4.2× worse (Ф8)
- **Parameters:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, SIGReg λ=0.09
- **Facts:** Ф8, Ф9, Ф10
- **Code:** paper2_full_analysis.py, drift_analysis_standalone.py, covariance_analysis_standalone.py
- **Data:** all_results.json (142KB)

### E07. Freeze test (Push-T, gym-pusht)
- **Environment:** Push-T (gym-pusht)
- **Conditions:** Free encoder frozen at epoch T = {1, 2, 3, 5, 7, 10}
- **Metric:** Best val loss
- **Result:** freeze@1: +20%, freeze@10: −1.1%. Causal evidence that drift is harmful. (Ф11)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф11
- **Code:** freeze_test_standalone.py
- **Data:** all_results.json

### E08. Random fixed encoder control (Push-T, synthetic)
- **Environment:** Push-T (synthetic physics)
- **Conditions:** Prescribed, rotated prescribed, random fixed 5→3, free
- **Metric:** Best val loss
- **Result:**
  - Random fixed 17× better than free (Ф12)
  - Prescribed 13× better than random fixed (Ф13)
  - Rotated ≈ prescribed at 1.09× (Ф14)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф12, Ф13, Ф14
- **Code:** random_fixed_encoder.py
- **Data:** random_fixed_v2_results.json (39KB)

### E09. Aligned-but-drifting + 2×2 factorial (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed, random fixed, aligned-drifting (linear + MLP), free
- **Metric:** Best val loss
- **Result:**
  - Aligned-drifting ≈ free or worse (Ф15)
  - 2×2 factorial: stability × alignment interaction 19× (Ф16)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф15, Ф16
- **Code:** paper2_aligned_drifting_colab.ipynb
- **Data:** aligned_drifting_results.json (49KB)

### E10. LR sweep + EMA baseline (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Free LR={1e-4, 3e-4, 1e-3, 3e-3}, free+EMA (decay=0.996), prescribed
- **Metric:** Best val loss, R²(0→1)
- **Result:** Prescribed wins at every LR (4.3–7.0×). EMA is 6.1× worse than prescribed.
- **Parameters:** Seed 42, 50 epochs
- **Facts:** (Paper 2, Section 5.6–5.7)
- **Code:** lr_sweep_ema_baseline.ipynb
- **Data:** lr_sweep_results.json (104KB)

### E11. Rico UI drift analysis (vision)
- **Environment:** Rico dataset, 398 UI screenshots
- **Conditions:** Free 3D + SIGReg vs ShovJEPA prescribed 3D
- **Metric:** R² transfer, effective rank, condition number
- **Result:** Drift in vision is weaker (R² 0.93 vs 0.78 in Push-T). Cross-modal confirmation.
- **Parameters:** Seed 42, 100 epochs
- **Facts:** (Paper 2, Section 5.8)
- **Code:** rico_drift_v2.ipynb
- **Data:** rico_drift_v2_results.json (53KB)

---

## Dim sweep experiments

### E12. 11 prescribed axes (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed_3, prescribed_11, free_3, free_11, random_fixed_11
- **Metric:** Val loss
- **Result:** prescribed_11 is 20× worse than prescribed_3. free_11 beats prescribed_11 by 6×. (Ф17)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф17
- **Code:** dim-sweep/exp1_11axes/run_11axes.py
- **Data:** dim-sweep/exp1_11axes/results.json

### E13. Dimension sweep 3–15 (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed vs free at dim = 3, 4, 5, 6, 7, 9, 11, 15
- **Metric:** Val loss
- **Result:** Crossover at dim=3→4. Prescribed wins only at dim ≤ 3. (Ф18)
- **Parameters:** 3 seeds, 20 epochs, 100 episodes (preliminary)
- **Facts:** Ф18
- **Code:** dim-sweep/exp2_sweep/run_sweep.py
- **Data:** dim-sweep/exp2_sweep/sweep_results.json

### E14. Lower boundary dim 1–3 (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed vs free at dim = 1, 2, 3
- **Metric:** Val loss
- **Result:** dim=1: 78×, dim=2: 12×, dim=3: 1.5×. Maximum advantage at minimum dim.
- **Parameters:** 2–3 seeds, 15–20 epochs, 50–100 episodes
- **Facts:** (included in Ф18)
- **Code:** dim-sweep/exp3_lower/run_lower.py
- **Data:** dim-sweep/exp3_lower/output.txt

### E15. Simple pendulum sweep (2 DOF)
- **Environment:** Simple pendulum (θ, θ̇), synthetic
- **Conditions:** Prescribed vs free at dim = 1–5, identical input
- **Metric:** Val loss
- **Result:** Free wins at all dims. No crossover. (Ф19)
- **Parameters:** 3 seeds, 20 epochs, 100 episodes
- **Facts:** Ф19
- **Code:** dim-sweep/exp4_pendulum/run_pendulum.py
- **Data:** dim-sweep/exp4_pendulum/results/

### E16. Double pendulum sweep (4 DOF)
- **Environment:** Double pendulum (θ₁, ω₁, θ₂, ω₂), synthetic
- **Conditions:** Prescribed vs free at dim = 1, 2, 4, 8, identical input
- **Metric:** Val loss
- **Result:** Prescribed wins only at dim=1 (2.1×). Free wins at dim ≥ 2. (Ф20)
- **Parameters:** 3 seeds, 20 epochs, 100 episodes
- **Facts:** Ф20
- **Code:** dim-sweep/exp5_double_pendulum/run_double_pendulum.py
- **Data:** dim-sweep/exp5_double_pendulum/results/results.json

### E17. Fragility test: types of 4th axis (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed_3, +sin(θ), +agent_x, +distance, +noise
- **Metric:** Val loss
- **Result:** Noise: 1106×. agent_x: 7.9×. Distance: 8.2×. sin(θ): 4.8×. (Ф21–Ф23)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф21, Ф22, Ф23
- **Code:** dim-sweep/exp6_fragility/run_fragility.py
- **Data:** dim-sweep/exp6_fragility/results/results.json

---

## Tier 1: Critical hypothesis tests

### E18. MLP decoder transfer (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Free encoder; at each epoch transition, train a linear and an MLP decoder on epoch t and evaluate on epoch t+1
- **Metric:** R² transfer (linear vs MLP)
- **Result:**
  - Epoch 0→1: MLP xfer = −283, linear xfer = −71 → information destroyed (Ф24)
  - Epoch 2+: MLP xfer ≈ 0.81, linear xfer ≈ 0.69 → information preserved, linear readability lost (Ф25)
  - The two-phase drift model is confirmed
- **Parameters:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes
- **Facts:** Ф24, Ф25
- **Code:** tier1_all_tests.py (T1 section)
- **Data:** tier1_results.json (T1 key)

### E19. Update ratio + differential LR (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed; free K=1,3,5 (predictor steps per encoder step); diffLR 10×, 100×
- **Metric:** Best val loss
- **Result:**
  - diffLR 100×: gap 62× (vs a 222× baseline) → 72% improvement, but 62× remains (Ф26)
  - K=3: WORSE than K=1 (−26%)
  - NOT pure optimization lag
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф26
- **Code:** tier1_all_tests.py (T2 section)
- **Data:** tier1_results.json (T2 key)

### E20. PCA canonicalization (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Free encoder; at each epoch, PCA-align the embeddings and measure R² transfer in canonical vs raw space
- **Metric:** R² transfer (raw vs PCA-canonical)
- **Result:** PCA worsens R² transfer at most epochs. The drift is non-linear, not rotation/scaling. (Ф27)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф27
- **Code:** tier1_all_tests.py (T3 section)
- **Data:** tier1_results.json (T3 key)

---

## Tier 2: Confound tests

### E21. Aligned-drifting ± SIGReg (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Aligned-linear ± SIGReg, free ± SIGReg, prescribed
- **Metric:** Best val loss
- **Result:**
  - SIGReg stabilizes aligned-linear (prevents divergence on seed 123) (Ф29)
  - Neither with nor without SIGReg does it approach prescribed (356× / 12601×)
  - Free without SIGReg is slightly better (0.007 vs 0.008)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф29
- **Code:** tier2_confound_tests.py (T4 section)
- **Data:** tier2_results.json (T4 key)

### E22. Optimizer state preservation in the freeze test (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** freeze@1 and @3 with a new optimizer vs preserving optimizer state
- **Metric:** Best val loss
- **Result:**
  - freeze@1: new_opt 0.008785, keep_state 0.008701 → difference 1.0% (Ф30)
  - freeze@3: difference 2.9%
  - The optimizer reset is NOT a confound
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф30
- **Code:** tier2_confound_tests.py (T5 section)
- **Data:** tier2_results.json (T5 key)

### E23. Random projection into a 3D vs 5D subspace (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed, rotated_prescribed, random_fixed_3d (block coords), random_fixed_5d (all coords), free
- **Metric:** Best val loss
- **Result:**
  - random_fixed_3d ≈ prescribed ≈ rotated_prescribed (all ~0.000037) (Ф31)
  - random_fixed_5d EXPLODES (376,053 mean) (Ф32)
  - Alignment within the subspace is irrelevant; subspace selection + normalization + freeze is the mechanism
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф31, Ф32
- **Code:** tier2_confound_tests.py (T7 section)
- **Data:** tier2_results.json (T7 key)

---

## Tier 3: Generalization tests

### E24. Baseline 3D comparison (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed 3D vs free 3D (same architecture as Tier 1–2)
- **Metric:** Best val loss, drift_01, R² transfer
- **Result:** Gap 169×. Drift 1.53. R² transfer −70.
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** (included in Ф33)
- **Code:** tier3_highdim.py (baseline section)
- **Data:** tier3_results.json (baseline_3d key)

### E25. 5D latent space (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed 5D (all 5 coords normalized), free MLP 5→5, random fixed orthogonal 5→5
- **Metric:** Best val loss, drift_01, R² transfer
- **Result:**
  - Gap prescribed/free: 66× (Ф33)
  - random_fixed ≈ prescribed (0.92×) (Ф34)
  - Drift 1.91, R² transfer −65
  - Prescribed without subspace selection still works (Ф36)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф33, Ф34, Ф36
- **Code:** tier3_highdim.py (T9a section)
- **Data:** tier3_results.json (T9a_5d key)

### E26. 16D latent space (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed 16D (engineered non-linear features), free MLP 5→16, random fixed 5→16
- **Metric:** Best val loss, drift_01, R² transfer
- **Result:**
  - Gap prescribed/free: 50× (Ф33)
  - random_fixed / prescribed: 1.53× — alignment begins to matter at high dim (Ф34)
  - Drift 3.58, R² transfer −596 — drift amplifies with dimension (Ф35)
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф33, Ф34, Ф35
- **Code:** tier3_highdim.py (T9b section)
- **Data:** tier3_results.json (T9b_16d key)

---

## Auxiliary: drift rate correlation

### E27. T8: drift rate vs downstream quality (Push-T, gym-pusht)
- **Environment:** Push-T (gym-pusht, real physics)
- **Conditions:** Existing data from E06 — no new training
- **Metric:** Pearson/Spearman correlation of drift rate × val loss, phase analysis
- **Result:**
  - Pearson = 0.95, Spearman = 0.51 (a non-linear relationship) (Ф28)
  - Two regimes: catastrophe (drift > 0.3) and saturation (drift < 0.1)
  - R² ceiling ≈ 0.75 (linear decoder on the free encoder)
  - Early drift is 8–14× larger than late drift
- **Parameters:** 3 seeds, 30 epochs, 200 episodes (data from E06)
- **Facts:** Ф28
- **Code:** analysis script (in-conversation)
- **Data:** all_results.json (from E06)

---

## П2 resolution

### E28. Dim sweep at full parameters (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed vs free at dim = 1, 2, 3, 4, 5, 7, 11. Predictor hidden = max(128, dim×8).
- **Metric:** Best val loss
- **Result:**
  - **NO CROSSOVER.** Prescribed wins at ALL dimensions 1–11.
  - dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
  - The gap decreases monotonically with dim but never reaches 1×
  - dim=5 matches Tier 3 E25 exactly (66.3× vs 66.2×)
  - **Ф18 (crossover at dim=4) REFUTED** — it was an underpowered artefact of E13
  - **Ф17 updated:** prescribed_11 now beats free_11 (42×) given proper predictor capacity
- **Parameters:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, predictor max(128, dim×8)
- **Facts:** Ф18 (refuted), Ф17 (updated)
- **Code:** p2_dim_sweep_full.py
- **Data:** p2_dim_sweep_results.json

### E29. Noise control: prescribed + matched noise vs free (Push-T, synthetic)
- **Environment:** Push-T (synthetic)
- **Conditions:** prescribed, prescribed+noise (i.i.d. late/mid/early/schedule), prescribed+correlated noise (mid/schedule), free
- **Metric:** Best val loss
- **Result:**
  - i.i.d. noise early (851×) is WORSE than free (222×) → drift ≠ random noise
  - Correlated noise early (1.3×) is FAR better than free (222×) → drift ≠ constant shift
  - The free encoder sits between i.i.d. and correlated → a data-dependent deformation
  - Spectrum: prescribed (1×) < correlated (1.3×) < noise_mid (6.2×) < FREE (222×) < noise_early (851×)
- **Parameters:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes
- **Facts:** Ф43i, Ф44i, Ф45i, Ф46i
- **Code:** noise_control.py
- **Data:** noise_control_results.json

---

## Drift / hallucination branch (Paper 2 line, 03.07.2026)

### E30. Critical window localization (Push-T, gym-pusht)
- **Environment:** Push-T (gym-pusht, real pymunk, synthetic=false)
- **Conditions:** Analysis of the freeze@k profile from E06/E07 all_results.json, no new runs. freeze@0 = random_fixed proxy (Ф12 = 0.000476), freeze@k for k ∈ {1,2,3,5,7,10}, unfrozen, prescribed.
- **Metric:** best_vp ratio between freeze points
- **Result:**
  - freeze@0 → freeze@1 = **136× cliff**; freeze@1 → unfrozen = 1.3× → ~99% of the damage in the first epoch (Ф45)
  - The freeze@k≥1 band is narrow (0.065–0.082), each point ≥25× worse than prescribed
  - Mechanistically closes Ф31 (random_fixed ≈ prescribed = freeze@0)
  - Incidentally: Г16 (drift-rate law) REFUTED on the same data (finding_drift_rate.docx) — the drift burns out in ~3 epochs, there is no constant rate
- **Parameters:** 3 seeds (42,123,777), 30 epochs, 200 episodes. Analysis, CPU, seconds. Reproduces exactly (136.25×).
- **Facts:** Ф45 (candidate → into Г18), Г16 (refuted)
- **Code:** E30_critical_window/code/analyze_critical_window.py
- **Data:** E30_critical_window/results/critical_window_results.json (+ finding_drift_rate.docx)

### E31. Sub-epoch freeze sweep (Push-T, synthetic)
- **Environment:** Push-T (synthetic, synth())
- **Conditions:** Freezing the free encoder at fractions of the batches of epoch 1: f ∈ {0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0}. The shape question: threshold or slope?
- **Metric:** best_vp vs freeze fraction; verdict from analyze_shape.py (linear-vs-step primary)
- **Result:**
  - **Verdict: SLOPE, not a threshold.** Over the rise f≥0.25 the line beats the best step by 2.2× (SS 0.085 vs 0.186), linear R²=0.88 (log space)
  - Monotone in 3/5 seeds (0 dips), 2/5 with a single noise dip; there is no sharp jump
  - The first quarter of the epoch is near-harmless; after that the damage integrates continuously
- **Parameters:** 5 seeds (42,123,777,7,99), 20 epochs, 100 episodes. CPU, ~45 runs, resume-safe.
- **Facts:** Ф46 (candidate)
- **Code:** E31_subepoch_freeze/code/subepoch_freeze.py, analyze_shape.py
- **Data:** E31_subepoch_freeze/results/subepoch_freeze_results.json, shape_verdict.txt

### E32. Sub-epoch freeze sweep on REAL data (Push-T, gym-pusht)
- **Environment:** Push-T (real gym-pusht, pymunk 6.2.1). A faithful port of freeze_test_standalone.py + collect_gym_data.
- **Conditions:** The E31 sweep on real physics. f ∈ {0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0}. To remove the synthetic caveat from Ф46.
- **Metric:** best_vp vs freeze fraction; verdict from analyze_shape.py
- **Result:**
  - **Verdict: SLOPE, cleaner than synthetic.** 5/5 seeds strictly monotone (0 dips)
  - Pooled linear-vs-step over the band f∈[0.25,0.60]: **linear R²=0.977**, step worse by **10.5×** (breakpoint f=0.50)
  - E30-style anchor freeze@1.0/@0.0 = 22.1× (per seed 7.2–35.3×) — the same direction as the 136× cliff
  - **Ф46 → SOLID** (the synthetic caveat is removed)
- **Parameters:** 5 seeds (7,42,123,777,2024), reduced budget EP=4/NEP=50 (sandbox limit; the shape is budget-robust). pymunk 6.2.1 pinned.
- **Caveats:** the absolute gaps are compressed (prescribed/unfrozen ~4× vs 222× at scale) — do NOT compare magnitudes with the 30-epoch runs. There is no sub-0.25 resolution (the near-harmless onset is qualitative). Per-seed raw seed_*.json are regenerated via run_seed.py.
- **Facts:** Ф46 (solid), Г18 (confirmed)
- **Code:** E32_subepoch_freeze_real/code/{e32_lib.py, run_seed.py, analyze_shape.py}
- **Data:** E32_subepoch_freeze_real/results/{shape_verdict.json, e32_slope.png}

---

---

## Pilot studies

### PreE30. Coordinate drift on DINOv2 (production-scale vision SSL)
- **Environment:** CIFAR-100 test split (random subset N=500), 32×32 → 224×224
- **Conditions:** facebook/dinov2-small (22M, 384D) vs facebook/dinov2-base (86M, 768D); CLS token from last_hidden_state; PCA equalization base→384D
- **Metric:** Procrustes R² (raw + Frobenius-normalized), linear CKA
- **Result:**
  - Procrustes R² (raw) = 0.645, R² (Frob-norm) = 0.650
  - Linear CKA (PCA-reduced) = 0.764, CKA (full dim) = 0.766
  - Sanity: R² self-comparison = 1.000 ✓; CKA vs Gaussian = 0.305 ⚠ (finite-sample artefact)
  - The pattern matches the drift hypothesis weakly: CKA exceeds R² by ~0.12. But the magnitude is moderate — production-scale pretraining (LVD-142M) stabilizes the coordinates considerably
- **Parameters:** N=500, seed=42, CPU only, a single sampling seed
- **Status:** PILOT_DONE. Not sufficient for standalone claims. To be removed once E36 is complete.
- **Limitations:** capacity confound (small ≠ base), N=500 is low power, CIFAR-100 is OOD for DINOv2, single seed, the CKA sanity check is broken
- **Code:** PreE30_drift_pilot_dinov2/code/e30_run.py
- **Data:** PreE30_drift_pilot_dinov2/results/results.json, embeddings.npz

### E36. Full coordinate drift on vision SSL (PLANNED)
- **Environment:** TBD (multi-seed fine-tune of DINOv2-small or equivalent)
- **Conditions:** Multi-seed runs (≥5 seeds), identical capacity, N≥5000, a correct shuffled-pairs baseline for CKA
- **Metric:** Procrustes R², linear CKA, error bars
- **Goal:** Close the capacity confound and the low N of PreE30; obtain standalone evidence for drift on production-scale vision SSL
- **Status:** PLANNED. Design: option A (fine-tune ImageNet-100, 5 seeds, ~5×3 H100-hours on RunPod) or option B (published multi-seed runs)
- **Supersedes:** PreE30 once complete

---

## Step 1 PCA diagnostic (Yadro Phase 2)

### E33. Step 1 PCA: last-token confound and pole stability on 5 LLMs
- **Environment:** Residual stream activations at the last token position; 5 LLMs (Qwen2.5-3B 36 layers, Gemma2-2B 26, OLMo-1B 16, Falcon-1B 22, Pythia-1.4B 32); 80 prompts × 8 categories from Step 0 yadro_phase2 (12 April 2026)
- **Conditions:** PCA(40), residualization by linear regression on the one-hot last token (41 unique tokens), LDA with 5-fold stratified CV
- **Metric:** R²(token) on the top-7 PCs, LDA on 7/15 PCs before and after residualization, normalized category entropy at PC poles, within-category cosine cohesion, Spearman/Pearson correlation, bootstrap stability (30 resamples at 70%), permutation null and leave-one-model-out for the Slice 1d association
- **Result:**
  - Path 2 (across layers): R²(token) falls monotonically with depth in all 5 LLMs (Δ=−0.24 to −0.32); LDA on 15 PCs after residualization rises monotonically (Δ=+0.18 to +0.25). Step 1 PCA was computed on middle layers in all 5 models (Qwen 18/36, Gemma 13/26, OLMo 8/16, Falcon 11/22, Pythia 16/32) — a consistent choice, not a Qwen-specific quirk
  - Slice 1: PAIRED PCs on residuals = 0 in all 5 models (1–4 on raw PCA). ASYMMETRIC PCs: 2–7
  - Slice 1b: distribution of asymmetric poles — code 8, emotional 7, abstract 4, factual 3, logical 2, spatial 1, narrative 0, ethical 0
  - Slice 1c: within-category cosine cohesion on final-layer residuals (mean over 5 models): emotional 0.128, spatial 0.086, code 0.086, abstract 0.083, factual 0.015, logical 0.003, narrative 0.000, ethical −0.019
  - Slice 1d: correlation of cohesion with the number of asymmetric appearances, Spearman ρ=0.826, Pearson r=0.746, N=8 under exact decomposition
  - Slice 1e: the permutation null (200 label shuffles, 163 valid) is centred near zero (mean −0.056, max 0.764) against an observed 0.826; p_perm = 0.006. LOO over the five models gives ρ 0.675–0.819 with no sign change. The code/spatial rank is decided by a gap of 1.267e−4 against a SEM of 0.012–0.018 — a tie, not an outlier
  - Slice 2: bootstrap stability of the nodes (≥70%) — only code in Falcon (1/5 models). No other category or model qualifies
- **Parameters:** All 5 models, every slice on the same activations from yadro_phase2; bootstrap seed=42, 30 resamples, PCA on 25 components; Slice 1e permutations seed=42
- **Status:** COMPLETED (diagnostic, not intervention)
- **Facts:** Ф47, Ф48, Ф49, Ф50, Ф51, Ф52, Ф53, Ф54, Ф55
- **Code:** E33_step1_pca_llm/code/{path2_layers,pc_polarities,srez1_polarity,srez1b_asym,srez1c_cohesion,srez1d_corr,srez1e_robustness,srez2_bootstrap,analiz_kod}.py
- **Data:** E33_step1_pca_llm/results/{path2,pc_polarities,srez1,srez1b,srez1c,srez1d,srez1e,srez2}_output.txt, path2_results.json, PC_polarities_on_residual.txt, srez3_pc_extremes_{qwen,gemma}.txt. PATH2_report.md was not committed to the repository (interpretation in Russian, outside the reproducibility requirements)
- **Limitations:** the Step 0 provenance is only partly established — `step1_pca_results.pkl` was computed outside the repository and does not record the layer index (the nearest candidate is Qwen layer 18, mean principal-angle cosine 0.907, with no exact match); at 80 samples the components above ~15 are not stably determined — neighbouring evr values differ in the 3rd–4th digit, and the choice of n_components=40 is arbitrary; until 21.08.2026 all ten PCA calls used `svd_solver='auto'` without a seed, which at this data shape selects randomized SVD — outputs produced before that date are not reproducible
- **Consequence for Step 2:** the current dataset is exhausted as a test of semantic geometry; controlled prompts are needed (equal length, a single shared last token), with activations taken from the final layer rather than a middle one

---

## EB-JEPA Two Rooms (transferring the prescribed approach to planning with obstacles)

### E34. EB-JEPA Two Rooms — prescribed_2 vs free planning
- **Environment:** EB-JEPA Two Rooms (Meta FAIR, 2602.03604), goal-conditioned navigation in a two-room environment with a vertical wall and a door. wall_x and door_y are randomized between trajectories (fix_wall=False)
- **Conditions:**
  - prescribed: PrescribedEncoder (MLP 2→256→256→512), input = the agent's (x_a, y_a), 199K params
  - free: ImpalaEncoder (CNN), input = 65×65 RGB pixels, 1.43M params
- **Shared:** RNNPredictor 793K params, regularizer = VICReg + IDM + temporal similarity, 12 epochs, batch=64, 100K episodes
- **Metric:** planning success rate (MPPI, 200 samples × 20 iter, plan_length=90, 200 steps), 20 episodes, epoch 11
- **Parameters:** seed=1 (single seed). 12 epochs. Default LeCun config
- **Result:**
  - free: SR = **55%** (11/20), mean_dist = 9.78
  - prescribed: SR = **0%** (0/20), mean_dist = 41.54
  - Probe loss: prescribed 0.006 (12× better than free at 0.072)
  - Pred loss: free 0.024 (2.2× better than prescribed at 0.051)
- **Status:** COMPLETED as a single-seed observation. **NOT closed as a fact under the programme protocol** (≥3 seeds required). The methodological gap (2D prescribed without information about the environment) is closed by E35
- **Platform:**
  - prescribed training — Colab Pro T4 GPU (~50h)
  - free training — Windows CPU, Python 3.14, PyTorch 2.11 (~167h)
  - planning eval — Colab Pro T4 GPU (~1.5h)
- **Observations:** Н1, Н2, Н3, Н4 (EVIDENCE.md)
- **Code:** E32_eb_jepa_planning/code/{eb_jepa_v3.ipynb, run_experiment_v3_windows.py, eb_jepa_planning_eval.ipynb}
- **Data:**
  - E32_eb_jepa_planning/results/{free,prescribed}/training_results.json (12 epochs each)
  - E32_eb_jepa_planning/results/{free,prescribed}/planning_eval_results.json
  - E32_eb_jepa_planning/results/{free,prescribed}/encoder_stats.json (12 epochs of aggregate stats — the first 20 of 512 dims)
  - latest.pth.tar checkpoints for both encoders (on Drive, in the archives eb_jepa_free.rar and eb_jepa_prescribed-*.zip)
  - plan_ep0/ — visualisations of 6 early planning episodes on free@ep0 (1 success, 5 fail)
- **Additional analysis (30.04.2026, post-hoc on encoder_stats.json):** the mean shift in latent space is larger for free relative to its norm than for prescribed (rel shift 0.69–0.78 vs 0.50–0.63). Caveat: these are aggregates over 20 of 512 dims after LayerNorm, not per-sample drift
- **Consequence for E35:** prescribed needs testing with environment coordinates (wall_x, door_y), not only the agent's

### E35. EB-JEPA Two Rooms — prescribed_4 (with wall and door coordinates)
- **Environment:** EB-JEPA Two Rooms, the same setup as E34
- **Condition:** prescribed_4 = (x_a, y_a, wall_x, door_y), z-score normalization (mean=[31.59, 32.06], std=[16.10, 16.14] — the same as Normalizer.normalize_location in the LeCun code)
- **Encoder:** PrescribedEncoder (MLP 4→256→256→512). Only the input dim differs from prescribed_2. Total ~199.5K params (comparable to prescribed_2's 199K)
- **Probe head:** stays 2D (x_a, y_a) for comparability with prescribed_2 and free
- **Parameters:** seed=1, 12 epochs, batch=64 — the same as E34 for a direct comparison
- **Metric:** planning success rate (same as E34), 20 episodes, epoch 11
- **Goal:** test Г22 (coordinate completeness as a condition for the prescribed advantage)
- **Falsifier:** SR ≈ 0% — the completeness hypothesis is refuted and the problem is deeper (architecture mismatch, insufficient capacity, a fundamental limitation)
- **Compute:** Windows CPU, ~60–100h in the background with auto-resume
- **Saving:** dual save — D:\experiments\E33_prescribed_4\results\ (local source of truth) + Drive backup (best-effort, mirrors only at the end of an epoch)
- **Dependencies:** none
- **Status:** READY_TO_START
- **Follow-up:**
  - If SR > 30%: a prescribed_3 = (x_a, y_a, wall_x) ablation — which matters more, the wall or the door
  - If SR ≈ 0%: a hybrid run (HybridEncoder is already in the code); a min-max normalization control on Г14
  - If SR is between 5–30%: another seed for variance estimation
- **Code:** run_experiment_v4_windows.py (in progress, in /home/claude/E35/)
- **Related documents:** work_plan_2026_04_30.md (the programme plan)

### E37. CARLA prescribed safety axes — physical axes in driving (DEFERRED)
- **Environment:** CARLA synthetic, 500 clips × 100 frames @ 10fps, 256×256
- **Conditions:** C1 free / C2 prescribed (4 axes: TTC, closing_v, lateral_offset, braking_margin) / C3 prescribed_frozen
- **Backbone:** V-JEPA 2.1 ViT-L (300M, frozen)
- **Metrics:** AP@(0.5s, 1s, 2s) lead time, R²(z_i, GT_i) per axis per epoch, eigenvalue spectrum stability
- **Compute:** ~6h of CARLA generation (Windows CPU) + ~45h Colab GPU (3 conditions × 3 seeds × 5h)
- **Status:** DEFERRED until E35 is complete
- **Dependencies:**
  - E35 finished (so that it is clear whether the prescribed approach survives on Two Rooms)
  - A stable Colab GPU (it currently falls back to CPU)
- **Code readiness:** 3 scripts in the archive (generate_carla_data.py, train.py, analyze.py), untested
- **Name:** **E37, not E16** (E16 is already taken by the double pendulum)

---
---

## Summary table

| ID | Name | Environment | Data | Seeds | Epochs | Episodes | Key result |
|---|---|---|---|---|---|---|---|
| E01 | Speech JEPA | LibriSpeech | — | pilot | — | — | +18–20pp entropy |
| E02 | Shov-JEPA vision | Rico UI | — | 1 | — | 398 | +5% accuracy |
| E03 | LeWM State | Push-T gym | gym | 3 | 50 | 200 | 38× |
| E04 | LeWM Pixel | Push-T pixel | gym | — | 50 | — | 14.8× |
| E05 | Controls (Paper 1) | Push-T | gym | 3 | 50 | 200 | random≈prescribed |
| E06 | Cov + drift | Push-T gym | gym | 3 | 30 | 200 | rank 2.99 → 233× worse |
| E07 | Freeze test | Push-T gym | gym | 3 | 30 | 200 | freeze@1 +20% |
| E08 | Random fixed | Push-T syn | syn | 3 | 30 | 200 | 17× stability |
| E09 | Aligned-drifting | Push-T syn | syn | 3 | 30 | 200 | aligned≈free |
| E10 | LR sweep + EMA | Push-T syn | syn | 1 | 50 | — | 4.3–7.0× at all LR |
| E11 | Rico drift | Rico UI | — | 1 | 100 | 398 | R²=0.93 |
| E12 | 11 axes | Push-T syn | syn | 3 | 30 | 200 | 20× worse |
| E13 | Dim sweep 3–15 | Push-T syn | syn | 3 | 20 | 100 | crossover 3→4 |
| E14 | Lower boundary | Push-T syn | syn | 2–3 | 15–20 | 50–100 | dim=1: 78× |
| E15 | Pendulum | Pendulum syn | syn | 3 | 20 | 100 | free always wins |
| E16 | Double pendulum | Dbl pend syn | syn | 3 | 20 | 100 | prescribed@dim=1 only |
| E17 | Fragility | Push-T syn | syn | 3 | 30 | 200 | noise: 1106× |
| E18 | MLP decoder xfer | Push-T syn | syn | 3 | 30 | 200 | ep0→1: info destroyed |
| E19 | Update ratio | Push-T syn | syn | 3 | 30 | 200 | 62× gap remains |
| E20 | PCA canonical | Push-T syn | syn | 3 | 30 | 200 | PCA worsens |
| E21 | ±SIGReg aligned | Push-T syn | syn | 3 | 30 | 200 | SIGReg stabilizes |
| E22 | Optimizer freeze | Push-T syn | syn | 3 | 30 | 200 | no confound |
| E23 | Random 3D vs 5D | Push-T syn | syn | 3 | 30 | 200 | random_3d≈prescribed |
| E24 | Baseline 3D | Push-T syn | syn | 3 | 30 | 200 | 169× |
| E25 | 5D latent | Push-T syn | syn | 3 | 30 | 200 | 66×, random≈prescribed |
| E26 | 16D latent | Push-T syn | syn | 3 | 30 | 200 | 50×, alignment emerges at 1.5× |
| E27 | Drift correlation | Push-T gym | gym | 3 | 30 | 200 | Pearson=0.95 |
| E28 | Dim sweep full | Push-T syn | syn | 3 | 30 | 200 | NO crossover, prescribed wins 1–11 |
| E29 | Noise control | Push-T syn | syn | 3 | 30 | 200 | drift≠noise, drift≠shift, free=222× |
| E30 | Critical window | Push-T gym | gym | 3 | 30 | 200 | 136× cliff: ~99% of the damage in epoch 1 |
| E31 | Sub-epoch freeze | Push-T syn | syn | 5 | 20 | 100 | SLOPE not a threshold (linear 2.2× best-step) |
| E32 | Sub-epoch freeze real | Push-T gym | gym | 5 | 4* | 50* | SLOPE, R²=0.977, 5/5 monotone, Ф46 solid |
| PreE30 | Drift pilot DINOv2 | CIFAR-100 | — | 1 | — | — | R²=0.65, CKA=0.77 (pilot) |
| E36 | Drift full DINOv2 | TBD | TBD | ≥5 | TBD | TBD | DEFERRED |
| E33 | Step 1 PCA diagnostic | LLM activations | yadro_phase2 | — | — | 80 prompts | last-token confound on 5 LLMs, nodes unstable |
| E34 | EB-JEPA Two Rooms prescribed_2 vs free | EB-JEPA Two Rooms | LeCun config 100K | 1 | 12 | 100K | free SR=55%, prescribed SR=0% (single-seed observation) |
| E35 | EB-JEPA Two Rooms prescribed_4 | EB-JEPA Two Rooms | LeCun config 100K | 1 | 12 | 100K | PLANNED — closing the E34 gap, testing Г22 |
| E37 | CARLA prescribed safety axes | CARLA synthetic | 500 clips | 3 | 30 | — | DEFERRED |


---

## Data files

| File | Experiments | Size | Source |
|---|---|---|---|
| all_results.json | E06, E07, E27 | 143KB | prescribed-axes-drift repo |
| random_fixed_v2_results.json | E08 | 39KB | prescribed-axes-drift repo |
| aligned_drifting_results.json | E09 | 49KB | prescribed-axes-drift repo |
| lr_sweep_results.json | E10 | 104KB | prescribed-axes-drift repo |
| rico_drift_v2_results.json | E11 | 53KB | prescribed-axes-drift repo |
| dim-sweep/exp1_11axes/results.json | E12 | 2KB | dim-sweep archive |
| dim-sweep/exp2_sweep/sweep_results.json | E13 | 3KB | dim-sweep archive |
| dim-sweep/exp3_lower/output.txt | E14 | <1KB | dim-sweep archive |
| dim-sweep/exp5_double_pendulum/results/results.json | E16 | 1KB | dim-sweep archive |
| dim-sweep/exp6_fragility/results/results.json | E17 | 1KB | dim-sweep archive |
| tier1_results.json | E18, E19, E20 | 30KB | Tier 1 script |
| tier2_results.json | E21, E22, E23 | 3KB | Tier 2 script |
| tier3_results.json | E24, E25, E26 | 5KB | Tier 3 script |
| p2_dim_sweep_results.json | E28 | 3KB | П2 resolution |
| noise_control_results.json | E29 | 8KB | E29 noise control (17.04.2026) |
| critical_window_results.json | E30 | 4KB | E30 critical window (03.07.2026) |
| finding_drift_rate.docx | E30 | — | Г16 refuted (03.07.2026) |
| subepoch_freeze_results.json | E31 | — | E31 sub-epoch (03.07.2026) |
| shape_verdict.txt | E31 | <1KB | E31 verdict SLOPE |
| shape_verdict.json | E32 | <1KB | E32 verdict SLOPE (03.07.2026) |
| e32_slope.png | E32 | — | E32 per-seed curves |
| seed_*.json | E32 | — | E32 per-seed raw (regenerated by run_seed.py) |

---

## Scripts

| File | Experiments | Source |
|---|---|---|
| paper2_full_analysis.py | E06, E07 | prescribed-axes-drift repo |
| random_fixed_encoder.py | E08 | prescribed-axes-drift repo |
| paper2_aligned_drifting_colab.ipynb | E09 | prescribed-axes-drift repo |
| lr_sweep_ema_baseline.ipynb | E10 | prescribed-axes-drift repo |
| rico_drift_v2.ipynb | E11 | prescribed-axes-drift repo |
| dim-sweep/exp1_11axes/run_11axes.py | E12 | dim-sweep archive |
| dim-sweep/exp2_sweep/run_sweep.py | E13 | dim-sweep archive (SUPERSEDED by E28) |
| dim-sweep/exp3_lower/run_lower.py | E14 | dim-sweep archive |
| dim-sweep/exp4_pendulum/run_pendulum.py | E15 | dim-sweep archive |
| dim-sweep/exp5_double_pendulum/run_double_pendulum.py | E16 | dim-sweep archive |
| dim-sweep/exp6_fragility/run_fragility.py | E17 | dim-sweep archive |
| tier1_all_tests.py | E18, E19, E20 | Tier 1 (15.04.2026) |
| tier2_confound_tests.py | E21, E22, E23 | Tier 2 (15.04.2026) |
| tier3_highdim.py | E24, E25, E26 | Tier 3 (15.04.2026) |
| p2_dim_sweep_full.py | E28 | П2 resolution (15.04.2026) |
| noise_control.py | E29 | E29 noise control (17.04.2026) |
| analyze_critical_window.py | E30 | drift-hallucination (03.07.2026) |
| subepoch_freeze.py + analyze_shape.py | E31 | drift-hallucination (03.07.2026) |
| e32_lib.py + run_seed.py + analyze_shape.py | E32 | drift-hallucination (03.07.2026) |

---

---

## Contradictions between experiments

**П1. E25 (5D prescribed works, 66×) vs E15/E16 (pendulum prescribed does not work)**
- Push-T 5D prescribed normalizes all coordinates → it works
- Pendulums: prescribed and free receive an identical input → it does not work
- Possibly: normalization, a difference in the dynamics, or the presence of "extra" (agent) coordinates in Push-T

**П2. ~~E13 (dim sweep: crossover at dim=4) vs E25 (prescribed_5d wins by 66×)~~ CLOSED**
- E28 (full parameters) confirms: NO crossover. Prescribed wins at dim 1–11.
- E13 was underpowered (100 ep, 20 epochs, 2 seeds, predictor hidden=128).
- dim=5 in E28 matches E25 exactly (66.3× vs 66.2×).

**П3. E08 (random_fixed_5d = 17× vs free) vs E23 (random_fixed_5d explodes)**
- E08: random_fixed projects 5→3, with a bias, on particular seeds
- E23: random_fixed projects 5→3, without input normalization, on different seeds
- A difference in implementation → a different result. The original E08 result is unreliable.


---

## Number mapping table (merge of 20.08.2026)

| April | New | What it is | Status |
|---|---|---|---|
| E30 | **E36** | Full coordinate drift on vision SSL | PLANNED |
| E31 | **E33** | Step 1 PCA, last-token confound, 5 LLMs | COMPLETED (Ф47–Ф55, Г19–Г24) |
| E32 | **E34** | EB-JEPA Two Rooms prescribed_2 vs free | COMPLETED as a single-seed observation (Н1–Н4) |
| E33 | **E35** | EB-JEPA Two Rooms prescribed_4 | READY_TO_START (Г25) |
| E34 | **E37** | CARLA prescribed safety axes | DEFERRED |

The July E30, E31, E32 and Г16, Г17, Г18 kept their numbers.
