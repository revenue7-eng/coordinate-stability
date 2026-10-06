# Experiment registry: Prescribed Axes

Author: Andrey Lazarev | Started: March 2026
Last updated: E50 recorded (VOID), registry audit Ф87, E01 to E04 renumbered to match the directories.

---

## Numbering

- **E01–E05**: Paper 1 (prescribed-axes). Numbered as the directories: E01 Shov-JEPA, E02 LeWM state, E03 LeWM pixel, E04 Speech JEPA.
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
- **E35**: EB-JEPA Two Rooms prescribed_4 — testing Г25 (COMPLETE 2026-09-09, see EVIDENCE Ф58). *Was E33.*
- **E36**: Full coordinate drift on vision SSL (PLANNED, see PreE30). *Was E30.*
- **E37**: CARLA prescribed safety axes (DEFERRED). *Was E34.*
- **E38**: Sub-epoch freeze sweep, full budget + sub-0.25 resolution (COMPLETE 2026-09-11, Ф46 revised, Ф60).
- **E39**: sub-epoch freeze micro-grid, five seeds (COMPLETE, Ф61, Ф62, Ф63).
- **E40**: initialisation sweep at a fixed data seed, frozen at step 0 (COMPLETE 2026-09-11, Ф64, Г26 split, Г27).
- **E41**: variance decomposition, encoder init x head init at a fixed data seed, frozen at step 0 (COMPLETE 2026-09-14, Ф65, Ф66, Ф67).
- **E42**: candidate-predictor sweep for Г27, four pre-registered candidates, Bonferroni 0.0125 (COMPLETE, Ф69 to Ф74).
- **E43**: external-target sweep over the E42 encoders, three pre-registered predictors, Bonferroni 0.0167 (COMPLETE, Ф75).
- **E44**: common-target comparison at the E28 dim-5 point (Ф77), COMPLETE: O4 PROVISIONAL (Ф78).
- **E45**: E44 at 90 epochs, prescribed against free_scaled (Ф78), COMPLETE: O4 PROVISIONAL (Ф79).
- **E46**: low data (Г-a) and the basis of the prescribed latent (Г-b), COMPLETE: Г-a refuted for plain prescribed (O5 PROVISIONAL at 25); standardising removes the early free advantage (Ф83).
- **E47**: standardised prescribed against free_scaled at 25, 50, 200 episodes, COMPLETE: O2 at every size, larger at 25 (Ф84). Reading revised by Н5: the prescribed arm is the state itself.
- **E48**: E47 on a nonlinear target, COMPLETE: O2 at 200 and 25 episodes (Ф85). Reading revised by Н5: the prescribed arm is the state itself.
- **E49**: completeness of the JEPA latent and an end-to-end learned encoder, COMPLETE: latent complete, verdict fixation with named limits (Ф86). Reading revised by Н5: the prescribed arm is the state itself.
- **E50**: fixation or geometry (fixed warped encoder; fair end-to-end learned encoder), COMPLETE: VOID, completeness gate failed (Н5).
- **E51+**: free. The nearest candidates are the n=30 replication of R2_readout on data seed 123 (Ф75 NEXT) and ECA / epiplexity (Г17).

> **Numbering collision (discovered 20.08.2026).** The April and July branches of the registry developed in parallel and independently used the numbers E30–E34 and Г16–Г22. The July numbers are committed in `648f1fd` and are referenced by the experiment READMEs and by Ф45/Ф46 — so it is the April branch that was renumbered. The mapping table is at the end of this file and in `EVIDENCE.md`.

---

## Paper 1: The Space Matters More Than the Loss

### E01. Shov-JEPA: 3 prescribed axes vs 64 free (Rico UI, vision)
- **Environment:** Rico dataset, 398 UI screenshots
- **Conditions:** ShovJEPA (3 axes: position, functionality, depth) vs free 64D
- **Metric:** Validation accuracy
- **Result:** 72.5% vs 67.5% (+5%)
- **Parameters:** 398 samples, single seed, pilot
- **Facts:** Ф4
- **Code:** prescribed-axes repo (shov-jepa)
- **Data:** shov-jepa-report-ru.docx

### E02. LeWM State: prescribed 3D vs free 3D (Push-T)
- **Environment:** Push-T (gym-pusht, pymunk physics)
- **Conditions:** Prescribed = normalize(x_b, y_b, θ_b) vs free MLP 5→3 + SIGReg
- **Metric:** Val prediction loss
- **Result:** Prescribed 0.004, free 0.157 = **38×**. Per axis: x 53×, y 63×, θ 25×
- **Parameters:** 3 seeds, 50 epochs, 200 episodes, SIGReg block
- **Facts:** Ф1
- **Code:** prescribed-axes repo (lewm_state)
- **Data:** lewm_state_results/

### E03. LeWM Pixel: prescribed 3D vs free CNN (Push-T from pixels)
- **Environment:** Push-T (96×96 pixel observations)
- **Conditions:** Prescribed 3D (20K params) vs free CNN (744K params)
- **Metric:** Val prediction loss
- **Result:** Prescribed **14.8×** better with **37× fewer parameters**. The CNN plateaus at epoch 7.
- **Parameters:** 50 epochs
- **Facts:** Ф2
- **Code:** prescribed-axes repo (lewm_pixels)
- **Data:** lewm_pixels_results/

### E04. Speech JEPA: prescribed cluster anchors vs free
- **Environment:** LibriSpeech
- **Conditions:** 2×2 factorial {GMM, k-means} × {soft, hard} vs pure JEPA
- **Metric:** Cluster entropy (codebook utilization)
- **Result:** +18–20pp entropy for prescribed. Soft ≈ hard (Δ<0.03%). Frozen structure is the dominant factor.
- **Parameters:** Pilot
- **Facts:** Ф3
- **Code:** prescribed-axes repo
- **Data:** —

### E05. Controls: random fixed, equal-input, SIGReg ablation (Push-T)
- **Environment:** Push-T
- **Conditions:** Random fixed 3D, free 3D same input, ±SIGReg
- **Metric:** Val prediction loss
- **Result:**
  - Random fixed ≈ prescribed (0.61×) → fixing > semantics (Ф5)
  - Equal-input free is 7.6× worse than prescribed → not about access to information (Ф6) [Ф82: own-latent ratio, not interpretable as a quality gap.]
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
  - 200 ep: random 0.61× prescribed, free 4.47× worse (Ф39) [Ф82: own-latent ratio, not interpretable as a quality gap.]
  - 500 ep: random 1.00× prescribed, **free 695,000× BETTER** (Ф38) [Ф80: the free latent is likely collapsed; this reading is withdrawn.]
  - Fixed encoders plateau at ~8.5×10⁻⁴, free → 10⁻⁹
  - Prescribed = sample efficiency, not absolute superiority
  - Isotropic normalization 15× worse (Ф40) [Ф80: rescaling of the metric latent, not evidence about the model.]
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
  - SIGReg harms free: 4.2× worse (Ф8) [Ф82: own-latent ratio, not interpretable as a quality gap.]
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
  - Random fixed 17× better than free (Ф12) [Ф82: own-latent ratio, not interpretable as a quality gap.]
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
- **Result:** Prescribed wins at every LR (4.3–7.0×). EMA is 6.1× worse than prescribed. [Ф82: own-latent ratio, not interpretable as a quality gap.]
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
- **Result:** prescribed_11 is 20× worse than prescribed_3. free_11 beats prescribed_11 by 6×. (Ф17) [Not interpretable as a quality gap: own-latent ratio, see Ф77, Ф80.]
- **Parameters:** 3 seeds, 30 epochs, 200 episodes
- **Facts:** Ф17
- **Code:** dim-sweep/exp1_11axes/run_11axes.py
- **Data:** dim-sweep/exp1_11axes/results.json

### E13. Dimension sweep 3–15 (Push-T)
- **Environment:** Push-T (synthetic)
- **Conditions:** Prescribed vs free at dim = 3, 4, 5, 6, 7, 9, 11, 15
- **Metric:** Val loss
- **Result:** Crossover at dim=3→4. Prescribed wins only at dim ≤ 3. (Ф18) [Not interpretable as a quality gap: own-latent ratio, see Ф77, Ф80.]
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
- **Result:** Free wins at all dims. No crossover. (Ф19) [Ф80: not interpretable.]
- **Parameters:** 3 seeds, 20 epochs, 100 episodes
- **Facts:** Ф19
- **Code:** dim-sweep/exp4_pendulum/run_pendulum.py
- **Data:** dim-sweep/exp4_pendulum/results/

### E16. Double pendulum sweep (4 DOF)
- **Environment:** Double pendulum (θ₁, ω₁, θ₂, ω₂), synthetic
- **Conditions:** Prescribed vs free at dim = 1, 2, 4, 8, identical input
- **Metric:** Val loss
- **Result:** Prescribed wins only at dim=1 (2.1×). Free wins at dim ≥ 2. (Ф20) [Ф80: not interpretable.]
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
  - **NO CROSSOVER.** Prescribed wins at ALL dimensions 1–11. [Not interpretable as a quality gap: own-latent ratio, see Ф77, Ф80.]
  - dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
  - The gap decreases monotonically with dim but never reaches 1×
  - dim=5 matches Tier 3 E25 exactly (66.3× vs 66.2×)
  - **Ф18 (crossover at dim=4) REFUTED** — it was an underpowered artefact of E13
  - **Ф17 updated:** prescribed_11 now beats free_11 (42×) given proper predictor capacity [Not interpretable as a quality gap: own-latent ratio, see Ф77, Ф80.]
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
- **Facts:** Ф41i, Ф42i, Ф43i, Ф44i
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
  - The freeze@k≥1 band is narrow (0.065–0.082), each point ≥25× worse than prescribed [Ф82: own-latent ratio, not interpretable as a quality gap.]
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

### E38. Sub-epoch freeze sweep — full budget + sub-0.25 resolution (Push-T, gym-pusht)
- **Environment:** Push-T (real gym-pusht, pymunk 6.2.1 pinned); infrastructure imported unchanged from E32 (`e32_lib.py`)
- **Conditions:** The E32 sweep at full budget with four added points below 0.25 and three between 0.60 and 1.00. f ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0}. Closes the three items E32 left open: forced budget, missing sub-0.25 resolution, absent per-seed raw data.
- **Metric:** best_vp per freeze fraction; shape by the E32 analyzer (linear vs single-breakpoint step), evaluated across several windows
- **Result:**
  - SLOPE over the rise f∈[0.00,0.40]: **5/5 seeds strictly monotone**, linear R²=0.880, step 1.86× worse
  - **Onset is not harmless**: f=0.00→0.25 gives 5.8–17.1× (Ф60), refuting the near-harmless reading in Ф46
  - Curve saturates rather than accelerating; per-seed plateau onset f≈0.70–1.00
  - The legacy 0.25–0.60 band returns STEP on this data — a window artifact, not a threshold (see the window table in the README)
  - Anchor freeze@1.0/@0.0 = 9.8–75.6× per seed (mean 39.1×), vs 22.1× at E32's reduced budget and 136× via E30's proxy
- **Parameters:** 5 seeds (42,123,777,2024,7), EP=15, NEP=200, 17 grid points. Local CPU, 33–267 min per seed (wall-clock varied with CPU contention, results unaffected).
- **Facts:** Ф60, Ф46 (revised)
- **Code:** E38_subepoch_freeze_full/code/{run_seed.py, analyze_shape.py, analyze_shape_windows.py}
- **Data:** E38_subepoch_freeze_full/results/seed_{7,42,123,777,2024}.json (checked in)
- **Code:** E32_subepoch_freeze_real/code/{e32_lib.py, run_seed.py, analyze_shape.py}
- **Data:** E32_subepoch_freeze_real/results/{shape_verdict.json, e32_slope.png}

---

---

### E39. Sub-epoch freeze micro-sweep, five seeds (Push-T, gym-pusht)
- **Status:** COMPLETE. Facts: Ф63. Registry: EVIDENCE.md. Nothing restated here.
- **Question:** how the frozen-at-step-0 encoder behaves across the opening
  window, and whether any cheap property of it tracks the outcome.
- **Design:** five seeds; initialisation and data sample vary together, which is
  the confound E40 was built to remove.
- **Left behind:** an observation that looked like a trend, the initialisation
  carrying the most linearly extractable information about the true state also
  gave the worst result. Unreadable as it stood, because the two factors moved
  together.
- **Code:** `E39_subepoch_freeze_micro/code/e39_lib.py` (copy of `e32_lib.py`).
- **Caveat for anyone comparing across experiments:** E39 step 0 never touched
  the RNG stream, E40 reseeds and restores it around model construction, so E39
  and E40 step-0 numbers are not directly comparable.

### E40. Initialisation sweep at a fixed data seed (Push-T, gym-pusht)
- **Status:** COMPLETE 2026-09-14. Facts: Ф64, Ф67, Ф68. Hypotheses: Г26 (split),
  Г27 (new). Registry: EVIDENCE.md.
- **Question:** with the data sample held fixed, how much does the frozen
  initialisation alone move `best_vp`, and does anything cheap order the
  initialisations?
- **Design:** ten initialisations, data seed fixed at 42, encoder frozen at step
  0 (`freeze_frac=0`) and never trained; only the predictor head trains.
  `R2_readout` and `eff_rank` measure what the fixed representation carries.
- **Metric:** `best_vp`, the minimum validation loss over epochs; lower is
  better.
- **Design note:** `e40_lib.py` adds an optional `init_seed`; the stream is
  reseeded immediately before the model is built and restored immediately after,
  so DataLoader shuffling order is identical across initialisations and only the
  parameters differ.
- **Code:** `E40_init_sweep/code/e40_lib.py` (copy of `e39_lib.py`).
- **Results:** `E40_init_sweep/results/sweep.json`.

### E41. Variance decomposition, encoder init x head init (Push-T, gym-pusht)
- **Status:** COMPLETE 2026-09-14. Facts: Ф65, Ф66, Ф67. Registry: EVIDENCE.md.
- **Question:** does the E40 spread belong to the encoder, or to the
  (encoder, head) pair? E40 drew both from one seed argument, so the two were
  confounded.
- **Design:** crossed grid, 8 encoder seeds x 5 head seeds, 40 cells, one run per
  cell, data seed 42 fixed, encoder frozen at step 0. Two-way crossed random
  effects, df 7 / 4 / 28, interaction confounded with the residual.
- **Acceptance:** encoder seeds 1..8 reproduce E40 initialisations 1..8
  bit-exactly, so the seed split did not move the stream.
- **Code:** `E41_variance_decomp/code/e41_lib.py`, `run_e41.py`,
  `analyze_e41.py`. **Results:** `E41_variance_decomp/results/grid.json`.

### E42. Candidate-predictor sweep for Г27 (Push-T, gym-pusht) — PRE-REGISTERED
- **Status:** PLANNED. Tests Г27 as refined 2026-09-14: the target is not a
  property that orders frozen bases in general, but one that explains the tails
  (Ф68).
- **Design:** 30 initialisations at data seed 42, one head each (justified by
  Ф66: the encoder level is recoverable from a single run, so no crossed design
  is needed). Encoder frozen at step 0. Plus 8-10 initialisations at a second
  data seed, run in the same campaign. The second data seed is 123, fixed
  2026-09-15 before any cell of the seed-42 half was read. It is one of the
  five E39 seeds, so E39 representation metrics exist on it for cross-check.
- **What the second data seed does and does not test:** it checks whether the
  spread and the variance shares reproduce off seed 42. It does NOT test any
  correlation: 8-10 points cannot resolve one.

**Candidate properties, closed list, all computed on the untrained encoder before
the first optimiser step, on the full validation split as returned by
`val_states(data_seed)` from E39's `analyze_representation.py`, which is the
same set of states behind Ф64 and Ф67. Not a batch: `eff_rank` and
`R2_readout` are imported from that module rather than reimplemented, so the
new values are the same quantities as 1.5380 and 1.5755. Candidate 4 needs the
window axis that `val_states` flattens away, so it reads the pre-reshape
tensor (Nv, H+2, 5); candidates 1 to 3 read the flattened one:**
1. `eff_rank` of the representation. Included for continuity with Ф67, where it
   was not a lead at n=8-10.
2. `R2_readout`, linear extractability of the true state. Included for continuity
   with Ф64, where corr = +0.060 at n=10.
3. Condition number of the representation covariance, defined as the ratio of
   largest to smallest eigenvalue of the same `np.cov(rep.T)` that `eff_rank`
   reads, so the two candidates are computed on one matrix. Distinct hypothesis from
   eff_rank: sensitive to the worst-conditioned direction rather than to how
   evenly variance is spread. Ill-conditioning is a plausible tail mechanism.
4. Smoothness of the latent dynamics:
   `mean(||z_{t+1} - z_t||) / sqrt(mean(||z_t||^2))` over consecutive states.
   The only candidate motivated by the task rather than by representation theory:
   the head predicts dynamics, and a representation in which the dynamics tear
   should be harder to predict in. Expected sign: positive with `best_vp`.
   Reachability checked before pre-registration: `DS` windows hold H+2 = 5
   consecutive states per item and `__getitem__` returns the window whole,
   so the pairs are within-sample and independent of the loader shuffle
   [e41_lib.py:57-66]. Windows are built at stride 1 and therefore overlap,
   so interior states enter the mean up to five times: the observations
   behind this candidate are not independent, and the jackknife report for
   it says so.

**Control, outside the multiplicity correction:** norm of the encoder output. It
scales the loss directly, so a correlation there would be about units rather than
about basis quality. Reported as a sanity check on whether SIGReg equalises scale
across initialisations; never counted as a hit.

**Statistics, fixed before the run:**
- Pearson and Spearman of each candidate against `best_vp` over all 30 points.
  AMBIGUITY FOUND AFTER THE RUN (2026-09-17): this line names two statistics
  and the next line one threshold, without saying which governs a hit. It
  decided nothing for candidates 1 to 3 and everything for candidate 4, which
  fails on Pearson (p = 0.0327) and passes on Spearman (p = 0.00088 Fisher,
  0.00127 permutation). Recorded here unresolved: choosing either one now
  would be choosing the test after seeing the result. A successor experiment
  names one statistic before it runs.
- Bonferroni over the four candidates: significance threshold alpha = 0.0125.
  At n=30, power 0.80, that resolves |rho| from 0.567 two-sided (uncorrected 0.492).
- Leave-one-out jackknife over the 30 points is reported for every candidate,
  significant or not. It is part of the report, not a response to an inconvenient
  result. Rationale: in this line the correlation has already been shown to be
  set by two points out of ten (Ф67, Ф68).
- The composition of points is not changed after seeing the results. Any subset
  analysis is reported alongside the full-sample figure, never in place of it.

**Declared outcomes:**
- If no candidate clears the corrected threshold: the property that orders frozen
  bases is not a cheap one. That is the registered result of this experiment, not
  a failure of it, and it closes Г27 negatively.
- If a candidate clears it and survives the jackknife: it becomes a lead and Г27
  moves to a confirmatory test on an independent data seed.
- If a candidate clears it and does not survive the jackknife: recorded as
  tail-driven, not as a lead, following the Ф67 precedent.
- Independently of the above, the 30 points measure the distribution of encoder
  levels, which is what the two-tier claim (Ф68) currently rests on with two
  observations out of ten.

**Cost:** about 86 s per cell as measured in E41, so roughly 45 min for the 30
plus about 15 min for the second-seed runs.

## Pilot studies

### E43. External-target sweep over the E42 encoders (Push-T, gym-pusht): PRE-REGISTERED
- **Status:** PLANNED. Written 2026-09-28 after Ф69 to Ф74 and before any
  external-target cell was run.
- **Question:** is the spread of E42 a property of the frozen encoders, or of
  the self-referential target? In E40 to E42 the head predicts emb[:, 3], the
  encoder's own output, so the encoder sets the scale of the loss it is scored
  by, and persistence orders the initialisations at +0.97 (Ф73). The declared
  E42 outcome "the property that orders frozen bases is not a cheap one" does
  not hold for that metric (Ф73).
- **Design:** the same frozen encoders as E42. Initialisation k at a data seed
  is built with enc_seed=k, head_seed=None, and every cell asserts that its
  frozen encoder equals the E42 checkpoint of the same cell tensor for tensor.
  The head is trained to predict PE(state at window position 3): block x/512,
  block y/512 and block angle/(2*pi) of the fourth state in the window, a fixed
  function of the true state that does not depend on the encoder. Context,
  action encoder, predictor, optimiser, epochs, split and batch order are those
  of E42. Code: `e43_lib.py` is `e42_lib.py` byte for byte plus an appended
  block that rebinds M; the first N bytes hash to c41d9434a5fc0ffc..., the
  sha256 of `e42_lib.py`.
- **Acceptance, already run:** in target "self", enc_seed=1 at data seed 42
  reproduced E40 init 1 bit for bit (best_vp 0.0031448905217346915,
  2026-09-28). That run produces no external-target output.
- **Cells:** 30 at data seed 42 (primary). 10 at data seed 123, which test
  reproduction of the spread and no correlation. 5 noise-floor cells: encoder 1
  at data seed 42 with head_seed 1..5.
- **Metric:** final_vp on the external target, on the log scale. Not best_vp,
  which carries selection over epochs (Ф74).

**Predictors, closed list, copied from E42 for the same encoder (Ф69, Ф73):**
1. log(persistence). The self-reference reading (Ф73) predicts no association:
   persistence ordered E42 through the scale of a target that is absent here.
2. R2_readout, linear extractability of the block pose from the frozen
   representation, which is the quantity the external head is asked to
   recover. The information reading predicts a negative association.
3. eff_rank. Predicted negative, from Ф74, which recorded it as the basis for
   this pre-registration.

Excluded, with reasons: cond_number correlates -0.7810 with eff_rank (Ф74);
smoothness is a function of rms_norm and persistence (Ф72); rms_norm was a
scale control in E42 and the external target has a fixed scale.

**Statistic, one, fixed before the run:**
- Pearson correlation of each predictor with log(final_vp) over the 30 cells at
  data seed 42; two-sided p from Fisher z. A hit is p < 0.0167 (Bonferroni over
  three). The sign is reported against the prediction; a hit with the
  unpredicted sign is recorded as a hit against the prediction. Spearman is
  printed alongside and decides nothing. At n=30 and power 0.80 the threshold
  resolves |r| from 0.553.
- Leave-one-out jackknife for every predictor, significant or not.
- The composition of points is not changed after seeing the results.
- The analysis is `E43_external_target/code/analyze_e43.py`, committed with
  this block. It prints the verdict below mechanically.

**Descriptive quantities with declared thresholds, outside the correction:**
- R = sd(log final_vp, E43) / sd(log final_vp, E42) over the same 30 encoders.
- Ordering agreement: Spearman between E43 final_vp and E42 final_vp over the 30.
- Noise floor F = sd(log final_vp) over the six heads on encoder 1 (head_seed
  None and 1..5) divided by sd(log final_vp, E43) over the 30 encoders.
- R at data seed 123 over its 10 encoders.

**Declared outcomes:**
- F >= 0.5: head noise is comparable to the encoder effect on this target.
  Correlations are reported and not interpreted, whatever their p.
- Collapse: R < 0.5 and predictor 1 is not a hit. The E42 spread is mainly a
  property of the self-referential target: comparing frozen encoders by a loss
  on their own output ranks them by latent step size.
- Persistence of the spread: R >= 0.5. The encoders differ on a target they do
  not define. A hit on predictor 2 or 3 with the predicted sign becomes a lead,
  to be confirmed in a separate experiment at n >= 30 on data seed 123. No hit
  means the difference is real and none of the three orders it.
- Predictor 1 a hit: persistence carries information about the encoder beyond
  the scale of the self-referential target. Recorded whatever R is.
- R at data seed 123 is reported next to R at data seed 42 and tests
  reproduction only.

**Cost:** E42 cells took 74.2 to 128.1 s on an unloaded machine (ce98ae4); the
E43 acceptance cell took 141 s. 45 cells: 60 to 106 min. Start only with no
concurrent build: cc1plus count 0 and load average under 1.

### E44. Common-target comparison at the E28 dim-5 point (synthetic Push-T): PRE-REGISTERED

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

### E45. E44 at 90 epochs, prescribed against free_scaled: PRE-REGISTERED

Question (Ф78). E44 classed free_scaled against prescribed as O4 but PROVISIONAL: both arms were still improving at epoch 30 (median convergence 0.863 and 0.873, below 0.95). Does the class hold when both stages are trained to convergence?

Setup. E44 unchanged except: arms prescribed and free_scaled only; 90 epochs in stage 1 and in stage 2. Library E44_common_target/code/e44_lib.py, unchanged since 483f39e; run_arm is called with epochs=90. Same seeds (42, 123, 777, 1001 to 1007), same data, split and loop. Workers run with OMP_NUM_THREADS=1, as E44 did; every seed file records the thread count.

Registered statistic and outcomes: as E44. Per seed L = ln(final2(free_scaled) / final2(prescribed)) at epoch 90; the class O1 to O5 from the 95% t interval (df 9) of GMR = exp(mean L) decides; the p value is reported only.

Convergence. conv = mean(stage-2 loss over epochs 81 to 90) / mean(epochs 71 to 80). If the median over seeds of conv for either arm is below 0.95, the class is reported as PROVISIONAL.

Validity gates, the campaign is void if any fails: prescribed stage 2 equals stage 1 bit for bit on every seed; the encoder state is unchanged by stage 2 in every cell; the first 30 epochs of stage 1 equal the E44 stage-1 history bit for bit for both arms on every seed (same code, data, seed and thread count, so any difference means the run is not what it claims to be); one pre-registration commit, ancestor of HEAD.

Named, not deciding: the registered statistic on best instead of final, with its class; per arm, E44 final at 30 epochs over E45 final at 90 epochs, as GMR with 95% interval (how much the extra training bought).

Scope: dim 5, E28 synthetic dynamics. If the class is O4 or O2 without PROVISIONAL, Ф78's reading stands as a programme fact for dim 5; if O1 or O3, Ф78 is corrected.

Code: E45_long_training/code/run_e45.py, analyze_e45.py. Results: E45_long_training/results/cells/seed_<s>.json, analysis.json.

Cost: E44 took 4 h 21 min wall for 720 epochs on the three-seed workers. E45 has 2 arms x 2 stages x 90 = 360 epochs per seed, 1080 on a three-seed worker: about 6.5 h wall on four workers.

Result (Ф79): O4 PROVISIONAL. GMR 0.915 [0.828, 1.010]; median convergence 0.916 and 0.940, below 0.95.

### E46. Low data and the basis of the prescribed latent (synthetic Push-T, E44 protocol): PRE-REGISTERED

Questions. Г-a: is a prescribed advantage on the common target visible at low data (25, 50 episodes), where the old claim of sample efficiency lived (Ф38 withdrawn, Ф80)? Г-b: in E45 stage 2, free_scaled learns faster than prescribed over the first epochs (Ф79 EXPLORATORY). Is that the basis of the prescribed latent: the wrap of theta/2pi (arm prescribed_sincos) or the small scale of range-normalised features (arm prescribed_std)? Rotation does not change it (Ф78, 1.00x), so only non-orthogonal changes are tested.

Code: E46_low_data_basis/code/run_e46.py imports E44_common_target/code/e44_lib.py unchanged; analyze_e46.py. Seeds 42, 123, 777, 1001 to 1007 (n = 10), as E44. 30 epochs per stage, OMP_NUM_THREADS=1 per worker.

Part A (Г-a). prescribed and free_scaled, both stages exactly as E44 run_arm, at EPISODES 25 and 50 (data synth(n, seed)). Fewer episodes at fixed epochs also means fewer optimiser steps, as in the old E05a design; the two are not separated here. Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed), GMR with 95% t interval (df 9), classes O1 to O5 as E44; PROVISIONAL if the median convergence ratio (E44 definition) is below 0.95. Г-a is supported if the class at 25 episodes is O1, O2 or O3; refuted if O4 or O5 at 25.

Part B (Г-b). At 200 episodes, stage 2 only (encoder fixed, fresh predictor on the common target, same seed and loop as E44 stage 2). Arms: prescribed_gate (plain prescribed); prescribed_sincos = (x_a, y_a, x_b, y_b)/512 and sin, cos of the angle (6 dims; the predictor output layer is 5 wide); prescribed_std = the five prescribed features standardised with mean and std of the seed's training split. Reference: E44 free_scaled stage 2 of the same seed. Registered statistic per arm: early = mean over stage-2 epochs 1 to 10 of ln(free_scaled / arm), GMR over seeds with 95% t interval. "Early free advantage removed" if the upper bound is at least 1, "present" if below 1. For E44 prescribed this is computed for reference. Final-epoch GMR and class are reported, not deciding.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; in part A prescribed stage 2 equals stage 1 bit for bit; the encoder is tensor-equal before and after every stage 2; prescribed_gate stage 2 equals E44 prescribed stage 2 bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. Г-a: 25 episodes O5 PROVISIONAL, 50 episodes O4 PROVISIONAL (plain prescribed); refuted as registered. Г-b: prescribed_std early free advantage removed, prescribed_sincos early free advantage present. See Ф83.
### E47. Standardised prescribed against free_scaled on the common target at 25, 50 and 200 episodes (synthetic Push-T): PRE-REGISTERED

Question. Ф83: with the prescribed features standardised, is there a prescribed advantage on the common target, and is it larger at low data?

Arms. prescribed_std: the five prescribed features standardised with mean and std of the training split of that seed and size; encoder fixed, stage 2 only (as E46 part B). free_scaled: both stages as E45, 90 epochs each, at 25 and 50 episodes; at 200 episodes taken from E45 cells (same seeds, data and code path). Common target and loop as E44. 90 epochs per stage. Seeds 42, 123, 777, 1001 to 1007 (n = 10). OMP_NUM_THREADS=1 per worker. Fewer episodes at fixed epochs also means fewer optimiser steps; not separated.

Code: E47_std_prescribed/code/run_e47.py (imports e44_lib.py unchanged), analyze_e47.py.

Registered statistic per size: L = ln(final2 free_scaled / final2 prescribed_std), GMR with 95% t interval (df 9), classes O1 to O5 as E44. Convergence: conv = mean(stage-2 loss, epochs 86 to 90) / mean(epochs 81 to 85); PROVISIONAL if the median over seeds for either arm is below 0.95. Г-g (a standardised fixed basis is better on the common target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5. Г-a' (the advantage is larger at low data): paired difference ln-ratio(25) minus ln-ratio(200) over seeds with 95% t interval; "larger at 25" if the lower bound is above 0, "smaller at 25" if the upper bound is below 0, otherwise not resolved.

Named, not deciding: GMR at epoch 60 against epoch 90 per size (ratio stability, Ф79 INTERPRETATION); the 50-episode class.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; on seed 42 plain prescribed stage 2 at 200 episodes and 90 epochs equals E45 prescribed s2_hist bit for bit; every E45 free_scaled history has 90 epochs.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. 200 episodes O2, 50 O2, 25 O2 PROVISIONAL; low-data difference larger at 25. Г-g and Г-a' supported as registered. See Ф84.
### E48. E47 on a target nonlinear in the prescribed coordinates (synthetic Push-T): PRE-REGISTERED

Question. Ф84 names the linear availability of the common target to prescribed_std as not excluded. Does the advantage of a standardised fixed basis survive when the target is nonlinear in the prescribed coordinates?

Target. g(s_{t+3}) = (d, sin theta, cos theta, u, v): d the agent-block distance, (u, v) the agent position relative to the block in the block's frame, positions divided by 512. Every component is nonlinear in the prescribed features. Stage 1 (own latent) is unchanged.

Arms and protocol as E47: prescribed_std (stage 2 only), free_scaled (both stages), 90 epochs per stage, at 25 and 200 episodes; free_scaled at 200 is rerun because its stage 2 target changes. Seeds 42, 123, 777, 1001 to 1007 (n = 10). Code: E48_nonlinear_target/code/run_e48.py (imports e44_lib.py unchanged; the world model's external target is replaced in a subclass), analyze_e48.py.

Registered statistic, classes, convergence rule and paired low-data difference exactly as E47. Г-i (the advantage survives a nonlinear target) is supported if the class at 200 episodes is O1, O2 or O3 and refuted if O4 or O5.

Named, not deciding: GMR on the nonlinear target divided by the E47 GMR at the same size (the share of the E47 advantage that the linear target carried); GMR at epoch 60.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the encoder is tensor-equal before and after every stage 2; free_scaled stage 1 at 200 episodes equals E45 free_scaled s1_hist bit for bit on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. 200 episodes O2, 25 O2 PROVISIONAL; low-data difference not resolved. Г-i supported as registered. See Ф85.
### E49. Completeness of the JEPA latent and an end-to-end learned encoder on the E48 target (synthetic Push-T): PRE-REGISTERED

Question. Ф85: a standardised fixed basis beats the frozen JEPA latent of free_scaled on a nonlinear target. Is that because the JEPA latent loses state information (Г-j, completeness), or because the coordinates are fixed (Г-k)?

Design, 200 episodes, 90 epochs, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. free_scaled stage 1 (JEPA on its own latent), identical to E45 and E48 stage 1. Its frozen latent is probed for the state at the same step, targets (x_a, y_a, x_b, y_b)/512, sin theta, cos theta: linear probe (least squares with bias) and MLP probe (5-64-64-6, 100 epochs, Adam 1e-3), fitted on the training split, R2 per target on the validation split. The same probes on prescribed_std are a sanity check.
2. free_e2e: FreeEncoderScaled trained end to end on the E48 target from the start (encoder not frozen, no JEPA stage), same loop and SIGReg weight.
prescribed_std and the frozen JEPA free_scaled on the same target are taken from E48 cells (same seeds, data and code path).
Code: E49_completeness/code/run_e49.py (imports e44_lib.py unchanged), analyze_e49.py.

Registered statistics. Completeness: per target the median over seeds of the MLP probe R2 on the JEPA latent; "incomplete" if the minimum over the six targets is below 0.95, "complete" if at least 0.99, otherwise unresolved. End to end: GMR free_e2e / prescribed_std on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: completeness (Г-j) if incomplete and free_e2e is O4 or O5; fixation (Г-k) if complete and free_e2e is O1, O2 or O3; otherwise mixed, reported as such.
Named, not deciding: linear probe R2; probes of the end-to-end encoder; GMR of frozen JEPA free_scaled over free_e2e.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; JEPA stage 1 equals E45 free_scaled s1_hist bit for bit on every seed; on prescribed_std the linear probe R2 of the four positions is at least 0.999 on every seed.

Cost: not estimated in advance; the monitor reports the measured rate.

Result: all gates true. Latent complete; free_e2e O2. Registered verdict: fixation, with two named limits. See Ф86.
### E50. Fixation or geometry: a fixed, nonlinearly warped, complete encoder on the E48 target (synthetic Push-T): PRE-REGISTERED

Question. Ф86: is the advantage of prescribed_std that its coordinates are fixed (Г-k) or that the state is laid out simply in them (Г-l)?

Arms, 200 episodes, E48 target, seeds 42, 123, 777, 1001 to 1007 (n = 10):
1. fixed_warped: frozen encoder f -> f Q1 -> h + A tanh(B h) -> Q2 -> h + A tanh(B h) -> Q3, then standardised on the training split; f the standardised prescribed features; A = 1.5, B = 2 (each step strictly monotone, so the map is invertible); Q1, Q2, Q3 random orthogonal from a private generator seeded 50000 + seed. Fixed like prescribed_std, complete, nonlinearly laid out like the learned latent. Stage 2 only, 90 epochs, E44 loop with SIGReg weight 0.09. Probed as in E49.
2. free_e2e_fair: FreeEncoderScaled trained end to end on the target for 180 epochs (the total budget of frozen JEPA), SIGReg weight 0, otherwise the E44 loop.
prescribed_std and frozen JEPA free_scaled are taken from E48 cells (same seeds, data, target).
Code: E50_fixation_vs_geometry/code/run_e50.py (imports e44_lib.py unchanged), analyze_e50.py.

Registered statistics: GMR fixed_warped / prescribed_std and GMR frozen JEPA (E48) / fixed_warped on the final epoch, 95% t interval, classes O1 to O5 as E44.
Verdict: fixation (Г-k) if fixed_warped / prescribed_std is O4 and frozen JEPA / fixed_warped is O1, O2 or O3; geometry (Г-l) if fixed_warped / prescribed_std is O1, O2 or O3 and frozen JEPA / fixed_warped is O4 or O5; otherwise mixed. The verdict is not deciding if the warp is too weak: minimum over the four positions of the median linear probe R2 of fixed_warped is at least 0.99.
Named, not deciding: GMR free_e2e_fair / prescribed_std and / frozen JEPA; linear and MLP probes of fixed_warped; median convergence ratios.

Gates, the campaign is void if any fails: every seed file carries this pre-registration commit; the fixed_warped encoder is tensor-equal before and after training; fixed_warped is complete (minimum median MLP probe R2 at least 0.99).

Cost: not estimated in advance; the monitor reports the measured rate.

Result: VOID, completeness gate failed (minimum median MLP probe R2 of fixed_warped 0.983 against 0.99). Exploratory numbers and the reading are in EVIDENCE Н5. Data: E50_fixation_vs_geometry/results/cells, analysis.json.

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
  - Probe loss: prescribed 0.006, free 0.072 — **not comparable across conditions**, the prescribed probe target is the encoder's own input (`run_experiment_v3_windows.py:490-494`). The 12× reading was withdrawn once the probe target was checked
  - Pred loss: free 0.024 (2.2× better than prescribed at 0.051)
- **Status:** COMPLETED as a single-seed observation. **NOT closed as a fact under the programme protocol** (≥3 seeds required). The methodological gap (2D prescribed without information about the environment) is closed by E35
- **Platform:**
  - prescribed training — Colab Pro T4 GPU (~50h)
  - free training — Windows CPU, Python 3.14, PyTorch 2.11 (~167h)
  - planning eval — Colab Pro T4 GPU (~1.5h)
- **Observations:** Н1, Н2, Н3, Н4 (EVIDENCE.md)
- **Code:** E34_eb_jepa_planning/code/{eb_jepa_v3.ipynb, run_experiment_v3_windows.py, eb_jepa_planning_eval.ipynb, b1_latent_physics.py, b1_control_prescribed.py}
- **Data:**
  - E34_eb_jepa_planning/results/{free,prescribed}/training_results.json (12 epochs each)
  - E34_eb_jepa_planning/results/{free,prescribed}/planning_eval_results.json
  - E34_eb_jepa_planning/results/{free,prescribed}/encoder_stats.json (12 epochs of aggregate stats — the first 20 of 512 dims)
  - latest.pth.tar checkpoints for both encoders (on Drive, in the archives eb_jepa_free.rar and eb_jepa_prescribed-*.zip)
  - plan_ep0/ — visualisations of 6 early planning episodes on free@ep0 (1 success, 5 fail)
- **Additional analysis (30.04.2026, post-hoc on encoder_stats.json):** the mean shift in latent space is larger for free relative to its norm than for prescribed (rel shift 0.69–0.78 vs 0.50–0.63). Caveat: these are aggregates over 20 of 512 dims after LayerNorm, not per-sample drift
- **B1, linear probe of the free latent:** ridge from the frozen 512d latent of the free checkpoint (seed 1, epoch 11) to the geometry, 4000 train / 1000 test fresh episodes, thresholds 0.7 / 0.3 fixed in advance. wall_x R² = 0.9691 (floor +0.0011 ± 0.0090), door_y R² = 0.2109 (floor -0.0038 ± 0.0034); t=8 gives 0.9687 and 0.2138. Structural control on prescribed_2: 0.0388 and 0.0336, against 0.0091 and 0.0147 from its own two-number input. Recorded as Ф56. Single checkpoint, no seed spread. Output in results/b1_output.txt and results/b1_control_output.txt
- **Consequence for E35:** prescribed needs testing with environment coordinates (wall_x, door_y), not only the agent's. B1 sharpens this: free itself uses wall_x heavily and door_y barely, so wall_x is the load-bearing addition and prescribed_3 = (x_a, y_a, wall_x) becomes a direct test of which coordinate carries the weight, rather than a follow-up conditional on SR > 30%

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
- **Status:** ON HOLD (24-25.08.2026) -> RESOLVED 2026-09-09 (see below). Planning SR at n=20 has no demonstrated power to separate an encoder holding wall_x (Ф56) from one holding nothing about the obstacle (Ф57). Resuming required a metric with established sensitivity and a quantitative falsifier; neither existed at hold time. See `E35_eb_jepa_prescribed4/README.md`.
- **RESOLVED (2026-09-09):** re-run at n=168 (vs the underpowered 20) now HAS power. Paired McNemar: prescribed_4 SR 0.577 vs free 0.470, diff +0.107, p=0.030 (significant). No pre-registered pass/fail threshold existed, so this is recorded as a significant positive effect, not a threshold verdict. See EVIDENCE Ф58. STATUS -> COMPLETE.
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

| ID | Name | Environment | Entries | Status |
|---|---|---|---|---|
| E01 | Shov-JEPA | Rico UI | Ф4 | pilot, 398 samples, one seed |
| E02 | LeWM state | Push-T | Ф1 | own-latent ratio, not interpretable (Ф82) |
| E03 | LeWM pixel | Push-T pixels | Ф2 | own-latent, not audited |
| E04 | Speech JEPA | LibriSpeech | Ф3 | pilot, not audited |
| E05 | Controls | Push-T | Ф5, Ф6, Ф7 | Ф5 rotation equality (Ф87); Ф6 not interpretable (Ф82) |
| E05a | Random axes scaling | Push-T | Ф38, Ф39, Ф40 | Ф38 collapse artefact; Ф39 rotation equality; Ф40 change of units (Ф80) |
| E05b | Gauge fixing | Push-T | Ф37 | not interpretable (Ф80) |
| E06 | Covariance + drift | Push-T | Ф8, Ф9, Ф10 | Ф9 not interpretable (Ф82); Ф10 movement stands |
| E07 | Freeze test | Push-T | Ф11 | not interpretable (Ф82) |
| E08 | Random fixed encoder | Push-T | Ф12, Ф13, Ф14 | different units (Ф87) |
| E09 | Aligned-but-drifting | Push-T | Ф15, Ф16 | not interpretable (Ф80, Ф82) |
| E10 | LR sweep + EMA | Push-T | Paper 2 | not interpretable (Ф82) |
| E11 | Rico drift | Rico UI | Paper 2 | movement observation |
| E12 | 11 axes | Push-T | Ф17 | not interpretable (Ф77) |
| E13 | Dim sweep 3 to 15 | Push-T | Ф18 | refuted, not interpretable (Ф77) |
| E14 | Lower boundary | Push-T | Ф18 | not interpretable (Ф77) |
| E15 | Pendulum | Pendulum | Ф19 | not interpretable (Ф80) |
| E16 | Double pendulum | Double pendulum | Ф20 | not interpretable (Ф80) |
| E17 | Fragility | Push-T | Ф21, Ф22, Ф23 | target variance, not damage (Ф87) |
| E18 | MLP decoder transfer | Push-T | Ф24, Ф25 | movement stands; information loss not supported (Ф87) |
| E19 | Update ratio + diffLR | Push-T | Ф26 | not interpretable (Ф82) |
| E20 | PCA canonicalisation | Push-T | Ф27 | stands |
| E21 | Aligned-drifting with SIGReg | Push-T | Ф29 | own-latent |
| E22 | Optimizer freeze | Push-T | Ф30 | stands |
| E23 | Random 3D vs 5D | Push-T | Ф31, Ф32 | rotation equality; units (Ф87) |
| E24 | Baseline 3D | Push-T | Ф33 | own-latent, not audited |
| E25 | 5D latent | Push-T | Ф33, Ф34, Ф36 | Ф36 corrected by Ф78 |
| E26 | 16D latent | Push-T | Ф33, Ф34, Ф35 | Ф35 movement stands; gaps own-latent |
| E27 | Drift correlation | Push-T | Ф28 | correlation with an own-latent loss |
| E28 | Dim sweep full | Push-T | Ф17, Ф18 | not interpretable (Ф77) |
| E29 | Noise control | Push-T | Ф41i to Ф44i | noise side stands (Ф87); free side not interpretable (Ф82) |
| E30 | Critical window | Push-T | Ф45 | own-latent (Ф82, Ф87) |
| E31 | Sub-epoch freeze | Push-T | Ф46 | own-latent target (Ф87) |
| E32 | Sub-epoch freeze real | Push-T gym | Ф46 | own-latent target (Ф87) |
| E33 | Step 1 PCA on LLMs | 5 LLMs | Ф47 to Ф55 | stands on one prompt set |
| E34 | EB-JEPA planning | Two Rooms | Ф56, Ф57 | both 0.55 SR on 20 episodes |
| E35 | EB-JEPA prescribed_4 | Two Rooms | Ф58, Ф59, Ф81 | 0.577 vs 0.470, n = 168, one seed; compares input modality |
| E36 | Vision SSL drift | DINOv2 | - | planned, not run |
| E37 | CARLA safety axes | CARLA | - | deferred, not run |
| E38 | Sub-epoch freeze full | Push-T gym | Ф46, Ф60 | own-latent target (Ф87) |
| E39 | Sub-epoch micro-grid | Push-T gym | Ф61, Ф62, Ф63 | Ф62 seeding bug stands |
| E40 | Initialisation sweep | Push-T gym | Ф64 | null corrected by Ф75 |
| E41 | Variance decomposition | Push-T gym | Ф65 to Ф68 | within an own-latent metric |
| E42 | Candidate sweep | Push-T gym | Ф69 to Ф74 | persistence orders frozen inits (Ф73) |
| E43 | External target | Push-T gym | Ф75 | R2_readout orders on an external target |
| E44 | Common target | synthetic Push-T | Ф76, Ф77, Ф78 | O4 PROVISIONAL |
| E45 | Long training | synthetic Push-T | Ф79 | O4 PROVISIONAL |
| E46 | Low data, basis | synthetic Push-T | Ф83 | scale of the fixed latent matters |
| E47 | Standardised prescribed | synthetic Push-T | Ф84 | state vs frozen JEPA latent (Н5) |
| E48 | Nonlinear target | synthetic Push-T | Ф85 | state vs frozen JEPA latent (Н5) |
| E49 | Completeness | synthetic Push-T | Ф86 | latent complete, laid out worse (Н5) |
| E50 | Fixation or geometry | synthetic Push-T | Н5 | VOID |
| PreE30 | DINOv2 drift pilot | CIFAR-100 | - | pilot |

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
- E28 (full parameters) confirms: NO crossover. Prescribed wins at dim 1–11. [Not interpretable as a quality gap: own-latent ratio, see Ф77, Ф80.]
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
| E33 | **E35** | EB-JEPA Two Rooms prescribed_4 | COMPLETE 2026-09-09 (Г25, Ф58) |
| E34 | **E37** | CARLA prescribed safety axes | DEFERRED |

The July E30, E31, E32 and Г16, Г17, Г18 kept their numbers.
