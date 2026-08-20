# Реестр экспериментов: Prescribed Axes

Автор: Andrey Lazarev | Дата начала: март 2026
Последнее обновление: 20 августа 2026 — слияние апрельской и июльской веток реестра.

---

## Нумерация

- **E01–E05**: Paper 1 (prescribed-axes)
- **E06–E12**: Paper 2 (prescribed-axes-drift)
- **E13–E18**: Dim sweep
- **E19–E21**: Tier 1 critical tests
- **E22–E24**: Tier 2 confound tests
- **E25–E27**: Tier 3 generalization tests
- **E28–E29**: П2 resolution + drift nature controls
- **PreE30**: Pilot — coordinate drift на DINOv2 (production-scale vision SSL)
- **E30–E32**: Drift / hallucination branch (critical window, sub-epoch freeze synthetic + real)
- **E33**: Step 1 PCA diagnostic — last-token confound и pole stability (LLM activations). *Был E31 в апрельской ветке.*
- **E34**: EB-JEPA Two Rooms prescribed_2 vs free — single-seed observation. *Был E32.*
- **E35**: EB-JEPA Two Rooms prescribed_4 — закрытие E34 gap, проверка Г25 (READY_TO_START). *Был E33.*
- **E36**: Full coordinate drift на vision SSL (PLANNED, см. PreE30). *Был E30.*
- **E37**: CARLA prescribed safety axes (DEFERRED). *Был E34.*
- **E38+**: свободны. Ближайший кандидат — ECA / epiplexity (Г17).

> **Коллизия нумерации (обнаружена 20.08.2026).** Апрельская и июльская ветки реестра развивались параллельно и независимо использовали номера E30–E34 и Г16–Г22. Июльские номера закоммичены в `648f1fd`, на них ссылаются README экспериментов и Ф45/Ф46 — поэтому перенумерована апрельская ветка. Таблица соответствия — в конце файла и в `EVIDENCE.md`.

---

## Paper 1: The Space Matters More Than the Loss

### E01. Speech JEPA: prescribed cluster anchors vs free
- **Среда:** LibriSpeech
- **Условия:** 2×2 factorial {GMM, k-means} × {soft, hard} vs Pure JEPA
- **Метрика:** Cluster entropy (codebook utilization)
- **Результат:** +18–20pp entropy для prescribed. Soft ≈ hard (Δ<0.03%). Frozen structure — доминирующий фактор.
- **Параметры:** Пилотное
- **Факты:** Ф3
- **Код:** prescribed-axes repo
- **Данные:** —

### E02. Shov-JEPA: 3 prescribed axes vs 64 free (Rico UI, Vision)
- **Среда:** Rico dataset, 398 UI screenshots
- **Условия:** ShovJEPA (3 axes: position, functionality, depth) vs Free 64D
- **Метрика:** Validation accuracy
- **Результат:** 72.5% vs 67.5% (+5%)
- **Параметры:** 398 samples, single seed, пилотное
- **Факты:** Ф4
- **Код:** prescribed-axes repo (shov-jepa)
- **Данные:** shov-jepa-report-ru.docx

### E03. LeWM State: prescribed 3D vs free 3D (Push-T)
- **Среда:** Push-T (gym-pusht, pymunk physics)
- **Условия:** Prescribed = normalize(x_b, y_b, θ_b) vs Free MLP 5→3 + SIGReg
- **Метрика:** Val prediction loss
- **Результат:** Prescribed 0.004, Free 0.157 = **38×**. Per-axis: x 53×, y 63×, θ 25×
- **Параметры:** 3 seeds, 50 epochs, 200 episodes, SIGReg block
- **Факты:** Ф1
- **Код:** prescribed-axes repo (lewm_state)
- **Данные:** lewm_state_results/

### E04. LeWM Pixel: prescribed 3D vs free CNN (Push-T from pixels)
- **Среда:** Push-T (96×96 pixel observations)
- **Условия:** Prescribed 3D (20K params) vs Free CNN (744K params)
- **Метрика:** Val prediction loss
- **Результат:** Prescribed **14.8×** лучше, **37× fewer parameters**. CNN плато на epoch 7.
- **Параметры:** 50 epochs
- **Факты:** Ф2
- **Код:** prescribed-axes repo (lewm_pixels)
- **Данные:** lewm_pixels_results/

### E05. Controls: random fixed, equal-input, SIGReg ablation (Push-T)
- **Среда:** Push-T
- **Условия:** Random fixed 3D, free 3D same input, ±SIGReg
- **Метрика:** Val prediction loss
- **Результат:**
  - Random fixed ≈ prescribed (0.61×) → фиксация > семантика (Ф5)
  - Equal-input free 7.6× хуже prescribed → не доступ к информации (Ф6)
  - SIGReg removal improves free 1.9× (Ф7)
- **Параметры:** 3 seeds, 50 epochs, 200 episodes
- **Факты:** Ф5, Ф6, Ф7
- **Код:** prescribed-axes repo (reviewer_response_experiments.py)
- **Данные:** reviewer_results/summary.json, reviewer_results/full_results.json

### E05a. Random axes scaling: 200ep vs 500ep (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, random_fixed, free_3d, free_5d at 200 and 500 episodes
- **Метрика:** Val prediction loss
- **Результат:**
  - 200 ep: random 0.61× prescribed, free 4.47× worse (Ф39)
  - 500 ep: random 1.00× prescribed, **free 695,000× BETTER** (Ф38)
  - Fixed encoders plateau at ~8.5×10⁻⁴, free → 10⁻⁹
  - Prescribed = sample efficiency, not absolute superiority
  - Isotropic normalization 15× worse (Ф40)
- **Параметры:** 200 ep: 3 seeds, 30 epochs. 500 ep: 3-9 runs, 50 epochs. No SIGReg.
- **Факты:** Ф38, Ф39, Ф40
- **Код:** random_axes_control/run_random_axes_control.py, run_isotropic_control.py
- **Данные:** exp5_random_axes/all_results.json (18 runs), random_fixed_results/results.json

### E05b. Gauge fixing free encoder (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, free, gauge_fixed_free, linear_free
- **Метрика:** Val prediction loss
- **Результат:**
  - gauge_fixed_free 1.08× ≈ free — gauge fixing не помогает (Ф37)
  - linear_free: 16009 (взрыв)
- **Параметры:** Data seed 42, training seeds [42, 123, 777], 50 epochs, synthetic
- **Факты:** Ф37
- **Код:** (gauge_fix experiment script)
- **Данные:** gauge_fix_results/results.json

---

## Paper 2: Semantic Drift, Not Rank Collapse

### E06. Covariance + Drift analysis (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht, real pymunk physics)
- **Условия:** Prescribed vs Free, covariance at sampled epochs, drift metrics
- **Метрика:** Effective rank, isotropy, raw/aligned drift, R² transfer
- **Результат:**
  - Free: rank 2.99, isotropy 0.86 → проигрывает prescribed 233× (Ф9)
  - R² transfer epoch 0→1: −16.9 / −62.2 / −25.4 (Ф10)
  - 80% drift structural after Procrustes
  - SIGReg вредит free: 4.2× хуже (Ф8)
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, SIGReg λ=0.09
- **Факты:** Ф8, Ф9, Ф10
- **Код:** paper2_full_analysis.py, drift_analysis_standalone.py, covariance_analysis_standalone.py
- **Данные:** all_results.json (142KB)

### E07. Freeze test (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht)
- **Условия:** Free encoder frozen at epoch T = {1, 2, 3, 5, 7, 10}
- **Метрика:** Best val loss
- **Результат:** Freeze@1: +20%, Freeze@10: −1.1%. Causal evidence for drift harm. (Ф11)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф11
- **Код:** freeze_test_standalone.py
- **Данные:** all_results.json

### E08. Random fixed encoder control (Push-T, synthetic)
- **Среда:** Push-T (synthetic physics)
- **Условия:** Prescribed, rotated prescribed, random fixed 5→3, free
- **Метрика:** Best val loss
- **Результат:**
  - Random fixed 17× лучше free (Ф12)
  - Prescribed 13× лучше random fixed (Ф13)
  - Rotated ≈ prescribed 1.09× (Ф14)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф12, Ф13, Ф14
- **Код:** random_fixed_encoder.py
- **Данные:** random_fixed_v2_results.json (39KB)

### E09. Aligned-but-drifting + 2×2 factorial (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed, random fixed, aligned-drifting (linear + MLP), free
- **Метрика:** Best val loss
- **Результат:**
  - Aligned-drifting ≈ free or worse (Ф15)
  - 2×2 factorial: stability × alignment interaction 19× (Ф16)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф15, Ф16
- **Код:** paper2_aligned_drifting_colab.ipynb
- **Данные:** aligned_drifting_results.json (49KB)

### E10. LR sweep + EMA baseline (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free LR={1e-4, 3e-4, 1e-3, 3e-3}, free+EMA (decay=0.996), prescribed
- **Метрика:** Best val loss, R²(0→1)
- **Результат:** Prescribed wins at every LR (4.3–7.0×). EMA 6.1× worse than prescribed.
- **Параметры:** Seed 42, 50 epochs
- **Факты:** (Paper 2, Section 5.6–5.7)
- **Код:** lr_sweep_ema_baseline.ipynb
- **Данные:** lr_sweep_results.json (104KB)

### E11. Rico UI drift analysis (Vision)
- **Среда:** Rico dataset, 398 UI screenshots
- **Условия:** Free 3D + SIGReg vs ShovJEPA prescribed 3D
- **Метрика:** R² transfer, effective rank, condition number
- **Результат:** Drift in vision weaker (R² 0.93 vs 0.78 in Push-T). Cross-modal confirmation.
- **Параметры:** Seed 42, 100 epochs
- **Факты:** (Paper 2, Section 5.8)
- **Код:** rico_drift_v2.ipynb
- **Данные:** rico_drift_v2_results.json (53KB)

---

## Dim Sweep Experiments

### E12. 11 prescribed axes (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed_3, prescribed_11, free_3, free_11, random_fixed_11
- **Метрика:** Val loss
- **Результат:** Prescribed_11 20× хуже prescribed_3. Free_11 beats prescribed_11 by 6×. (Ф17)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф17
- **Код:** dim-sweep/exp1_11axes/run_11axes.py
- **Данные:** dim-sweep/exp1_11axes/results.json

### E13. Dimension sweep 3–15 (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 3, 4, 5, 6, 7, 9, 11, 15
- **Метрика:** Val loss
- **Результат:** Crossover at dim=3→4. Prescribed wins only dim ≤ 3. (Ф18)
- **Параметры:** 3 seeds, 20 epochs, 100 episodes (предварительные)
- **Факты:** Ф18
- **Код:** dim-sweep/exp2_sweep/run_sweep.py
- **Данные:** dim-sweep/exp2_sweep/sweep_results.json

### E14. Lower boundary dim 1–3 (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 1, 2, 3
- **Метрика:** Val loss
- **Результат:** dim=1: 78×, dim=2: 12×, dim=3: 1.5×. Max advantage at min dim.
- **Параметры:** 2–3 seeds, 15–20 epochs, 50–100 episodes
- **Факты:** (включено в Ф18)
- **Код:** dim-sweep/exp3_lower/run_lower.py
- **Данные:** dim-sweep/exp3_lower/output.txt

### E15. Simple pendulum sweep (2 DOF)
- **Среда:** Simple pendulum (θ, θ̇), synthetic
- **Условия:** Prescribed vs free at dim = 1–5, identical input
- **Метрика:** Val loss
- **Результат:** Free wins at all dims. No crossover. (Ф19)
- **Параметры:** 3 seeds, 20 epochs, 100 episodes
- **Факты:** Ф19
- **Код:** dim-sweep/exp4_pendulum/run_pendulum.py
- **Данные:** dim-sweep/exp4_pendulum/results/

### E16. Double pendulum sweep (4 DOF)
- **Среда:** Double pendulum (θ₁, ω₁, θ₂, ω₂), synthetic
- **Условия:** Prescribed vs free at dim = 1, 2, 4, 8, identical input
- **Метрика:** Val loss
- **Результат:** Prescribed wins only dim=1 (2.1×). Free wins dim ≥ 2. (Ф20)
- **Параметры:** 3 seeds, 20 epochs, 100 episodes
- **Факты:** Ф20
- **Код:** dim-sweep/exp5_double_pendulum/run_double_pendulum.py
- **Данные:** dim-sweep/exp5_double_pendulum/results/results.json

### E17. Fragility test: 4th axis types (Push-T)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed_3, +sin(θ), +agent_x, +distance, +noise
- **Метрика:** Val loss
- **Результат:** Noise: 1106×. Agent_x: 7.9×. Distance: 8.2×. Sin(θ): 4.8×. (Ф21–Ф23)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф21, Ф22, Ф23
- **Код:** dim-sweep/exp6_fragility/run_fragility.py
- **Данные:** dim-sweep/exp6_fragility/results/results.json

---

## Tier 1: Critical Hypothesis Tests

### E18. MLP decoder transfer (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free encoder, each epoch transition: train linear + MLP decoder on epoch t, evaluate on epoch t+1
- **Метрика:** R² transfer (linear vs MLP)
- **Результат:**
  - Epoch 0→1: MLP xfer = −283, linear xfer = −71 → information destroyed (Ф24)
  - Epoch 2+: MLP xfer ≈ 0.81, linear xfer ≈ 0.69 → info preserved, linear readability lost (Ф25)
  - Two-phase drift model confirmed
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes
- **Факты:** Ф24, Ф25
- **Код:** tier1_all_tests.py (T1 section)
- **Данные:** tier1_results.json (T1 key)

### E19. Update ratio + differential LR (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed; Free K=1,3,5 (predictor steps per encoder step); diffLR 10×, 100×
- **Метрика:** Best val loss
- **Результат:**
  - DiffLR 100×: gap 62× (vs 222× baseline) → 72% improvement, but 62× remains (Ф26)
  - K=3: WORSE than K=1 (−26%)
  - NOT pure optimization lag
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф26
- **Код:** tier1_all_tests.py (T2 section)
- **Данные:** tier1_results.json (T2 key)

### E20. PCA canonicalization (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Free encoder, each epoch: PCA-align embeddings, measure R² transfer in canonical vs raw space
- **Метрика:** R² transfer (raw vs PCA-canonical)
- **Результат:** PCA worsens R² transfer at most epochs. Drift is nonlinear, not rotation/scaling. (Ф27)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф27
- **Код:** tier1_all_tests.py (T3 section)
- **Данные:** tier1_results.json (T3 key)

---

## Tier 2: Confound Tests

### E21. Aligned-drifting ± SIGReg (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Aligned-linear ± SIGReg, free ± SIGReg, prescribed
- **Метрика:** Best val loss
- **Результат:**
  - SIGReg stabilizes aligned-linear (prevents divergence on seed 123) (Ф29)
  - Neither with nor without SIGReg approaches prescribed (356× / 12601×)
  - Free without SIGReg slightly better (0.007 vs 0.008)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф29
- **Код:** tier2_confound_tests.py (T4 section)
- **Данные:** tier2_results.json (T4 key)

### E22. Optimizer state preservation in freeze test (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Freeze@1 and @3 with new optimizer vs preserving optimizer state
- **Метрика:** Best val loss
- **Результат:**
  - freeze@1: new_opt 0.008785, keep_state 0.008701 → difference 1.0% (Ф30)
  - freeze@3: difference 2.9%
  - Optimizer reset is NOT a confound
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф30
- **Код:** tier2_confound_tests.py (T5 section)
- **Данные:** tier2_results.json (T5 key)

### E23. Random projection 3D vs 5D subspace (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, rotated_prescribed, random_fixed_3d (block coords), random_fixed_5d (all coords), free
- **Метрика:** Best val loss
- **Результат:**
  - random_fixed_3d ≈ prescribed ≈ rotated_prescribed (all ~0.000037) (Ф31)
  - random_fixed_5d EXPLODES (376,053 mean) (Ф32)
  - Alignment within subspace irrelevant; subspace selection + normalization + freeze = the mechanism
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф31, Ф32
- **Код:** tier2_confound_tests.py (T7 section)
- **Данные:** tier2_results.json (T7 key)

---

## Tier 3: Generalization Tests

### E24. Baseline 3D comparison (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 3D vs Free 3D (same architecture as Tier 1-2)
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:** Gap 169×. Drift 1.53. R² transfer −70.
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** (included in Ф33)
- **Код:** tier3_highdim.py (baseline section)
- **Данные:** tier3_results.json (baseline_3d key)

### E25. 5D latent space (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 5D (all 5 coords normalized), Free MLP 5→5, Random fixed orthogonal 5→5
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:**
  - Gap prescribed/free: 66× (Ф33)
  - random_fixed ≈ prescribed (0.92×) (Ф34)
  - Drift 1.91, R² transfer −65
  - Prescribed without subspace selection still works (Ф36)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф33, Ф34, Ф36
- **Код:** tier3_highdim.py (T9a section)
- **Данные:** tier3_results.json (T9a_5d key)

### E26. 16D latent space (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed 16D (engineered nonlinear features), Free MLP 5→16, Random fixed 5→16
- **Метрика:** Best val loss, drift_01, R² transfer
- **Результат:**
  - Gap prescribed/free: 50× (Ф33)
  - random_fixed / prescribed: 1.53× — alignment begins to matter at high dim (Ф34)
  - Drift 3.58, R² transfer −596 — drift amplifies with dimension (Ф35)
- **Параметры:** 3 seeds, 30 epochs, 200 episodes
- **Факты:** Ф33, Ф34, Ф35
- **Код:** tier3_highdim.py (T9b section)
- **Данные:** tier3_results.json (T9b_16d key)

---

## Вспомогательный: Drift rate correlation

### E27. T8: Drift rate vs downstream quality (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht, real physics)
- **Условия:** Existing data from E06 — no new training
- **Метрика:** Pearson/Spearman correlation drift rate × val loss, phase analysis
- **Результат:**
  - Pearson = 0.95, Spearman = 0.51 (nonlinear relationship) (Ф28)
  - Two regimes: catastrophe (drift > 0.3) and saturation (drift < 0.1)
  - R² ceiling ≈ 0.75 (linear decoder on free encoder)
  - Early drift 8–14× larger than late drift
- **Параметры:** 3 seeds, 30 epochs, 200 episodes (data from E06)
- **Факты:** Ф28
- **Код:** analysis script (in-conversation)
- **Данные:** all_results.json (from E06)

---

## П2 Resolution

### E28. Dim sweep full parameters (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** Prescribed vs free at dim = 1, 2, 3, 4, 5, 7, 11. Predictor hidden = max(128, dim×8).
- **Метрика:** Best val loss
- **Результат:**
  - **NO CROSSOVER.** Prescribed wins at ALL dimensions 1–11.
  - dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
  - Gap monotonically decreases with dim but never reaches 1×
  - dim=5 matches Tier 3 E25 exactly (66.3× vs 66.2×)
  - **Ф18 (crossover at dim=4) REFUTED** — was underpowered artifact of E13
  - **Ф17 updated:** prescribed_11 now beats free_11 (42×) with proper predictor capacity
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes, predictor max(128, dim×8)
- **Факты:** Ф18 (refuted), Ф17 (updated)
- **Код:** p2_dim_sweep_full.py
- **Данные:** p2_dim_sweep_results.json

### E29. Noise control: prescribed + matched noise vs free (Push-T, synthetic)
- **Среда:** Push-T (synthetic)
- **Условия:** prescribed, prescribed+noise (i.i.d. late/mid/early/schedule), prescribed+correlated noise (mid/schedule), free
- **Метрика:** Best val loss
- **Результат:**
  - i.i.d. noise early (851×) ХУЖЕ free (222×) → drift ≠ random noise
  - Correlated noise early (1.3×) ЗНАЧИТЕЛЬНО лучше free (222×) → drift ≠ constant shift
  - Free encoder между i.i.d. и correlated → data-dependent deformation
  - Спектр: prescribed (1×) < correlated (1.3×) < noise_mid (6.2×) < FREE (222×) < noise_early (851×)
- **Параметры:** 3 seeds (42, 123, 777), 30 epochs, 200 episodes
- **Факты:** Ф43i, Ф44i, Ф45i, Ф46i
- **Код:** noise_control.py
- **Данные:** noise_control_results.json

---

## Drift / Hallucination branch (Paper 2 line, 03.07.2026)

### E30. Critical window localization (Push-T, gym-pusht)
- **Среда:** Push-T (gym-pusht, real pymunk, synthetic=false)
- **Условия:** Анализ freeze@k-профиля из E06/E07 all_results.json, без новых прогонов. freeze@0 = random_fixed прокси (Ф12 = 0.000476), freeze@k для k ∈ {1,2,3,5,7,10}, unfrozen, prescribed.
- **Метрика:** best_vp ratio между freeze-точками
- **Результат:**
  - freeze@0 → freeze@1 = **136× cliff**; freeze@1 → unfrozen = 1.3× → ~99% повреждения в первой эпохе (Ф45)
  - band freeze@k≥1 узкий (0.065–0.082), каждый ≥25× хуже prescribed
  - Механистически замыкает Ф31 (random_fixed ≈ prescribed = freeze@0)
  - Побочно: Г16 (drift-rate law) ОПРОВЕРГНУТ на тех же данных (finding_drift_rate.docx) — дрейф выгорает за ~3 эпохи, постоянного rate нет
- **Параметры:** 3 сида (42,123,777), 30 эпох, 200 эпизодов. Анализ, CPU, секунды. Воспроизводится точно (136.25×).
- **Факты:** Ф45 (candidate → в Г18), Г16 (refuted)
- **Код:** E30_critical_window/code/analyze_critical_window.py
- **Данные:** E30_critical_window/results/critical_window_results.json (+ finding_drift_rate.docx)

### E31. Sub-epoch freeze sweep (Push-T, synthetic)
- **Среда:** Push-T (synthetic, synth())
- **Условия:** Заморозка free-энкодера на долях батчей эпохи 1: f ∈ {0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 1.0}. Shape-вопрос: порог или склон?
- **Метрика:** best_vp vs freeze-fraction; verdict от analyze_shape.py (linear-vs-step primary)
- **Результат:**
  - **Verdict: SLOPE, не порог.** Над подъёмом f≥0.25 линия лучше best-step в 2.2× (SS 0.085 vs 0.186), linear R²=0.88 (log-space)
  - Монотонно 3/5 сидов (0 спадов), 2/5 по одному шумовому спаду; резкого скачка нет
  - Первая четверть эпохи near-harmless; далее ущерб интегрируется непрерывно
- **Параметры:** 5 сидов (42,123,777,7,99), 20 эпох, 100 эпизодов. CPU, ~45 прогонов, resume-safe.
- **Факты:** Ф46 (candidate)
- **Код:** E31_subepoch_freeze/code/subepoch_freeze.py, analyze_shape.py
- **Данные:** E31_subepoch_freeze/results/subepoch_freeze_results.json, shape_verdict.txt

### E32. Sub-epoch freeze sweep on REAL data (Push-T, gym-pusht)
- **Среда:** Push-T (real gym-pusht, pymunk 6.2.1). Верный порт freeze_test_standalone.py + collect_gym_data.
- **Условия:** E31-свип на реальной физике. f ∈ {0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0}. Снять synthetic-оговорку с Ф46.
- **Метрика:** best_vp vs freeze-fraction; verdict от analyze_shape.py
- **Результат:**
  - **Verdict: SLOPE, чище synthetic.** 5/5 сидов строго монотонны (0 спадов)
  - Pooled linear-vs-step над band f∈[0.25,0.60]: **linear R²=0.977**, step хуже в **10.5×** (breakpoint f=0.50)
  - E30-style anchor freeze@1.0/@0.0 = 22.1× (per seed 7.2–35.3×) — то же направление, что 136× cliff
  - **Ф46 → SOLID** (synthetic-оговорка снята)
- **Параметры:** 5 сидов (7,42,123,777,2024), reduced budget EP=4/NEP=50 (sandbox-предел; shape budget-robust). pymunk 6.2.1 pinned.
- **Оговорки:** абсолютные gap сжаты (prescribed/unfrozen ~4× vs 222× at scale) — магнитуды НЕ сравнивать с 30-эпоховыми. Sub-0.25 разрешения нет (near-harmless onset качественный). Per-seed raw seed_*.json регенерируются через run_seed.py.
- **Факты:** Ф46 (solid), Г18 (confirmed)
- **Код:** E32_subepoch_freeze_real/code/{e32_lib.py, run_seed.py, analyze_shape.py}
- **Данные:** E32_subepoch_freeze_real/results/{shape_verdict.json, e32_slope.png}

---

---

## Pilot studies

### PreE30. Coordinate drift on DINOv2 (production-scale vision SSL)
- **Среда:** CIFAR-100 test split (random subset N=500), 32×32 → 224×224
- **Условия:** facebook/dinov2-small (22M, 384D) vs facebook/dinov2-base (86M, 768D); CLS token из last_hidden_state; PCA equalization base→384D
- **Метрика:** Procrustes R² (raw + Frob-norm), Linear CKA
- **Результат:**
  - Procrustes R² (raw) = 0.645, R² (Frob-norm) = 0.650
  - Linear CKA (PCA-reduced) = 0.764, CKA (full dim) = 0.766
  - Sanity: R² self-comparison = 1.000 ✓; CKA vs Gaussian = 0.305 ⚠ (finite-sample artifact)
  - Pattern matches drift hypothesis weakly: CKA > R² на ~0.12. Но магнитуда умеренная — production-scale pretrain (LVD-142M) сильно стабилизирует координаты
- **Параметры:** N=500, seed=42, CPU only, single seed выборки
- **Статус:** PILOT_DONE. Не достаточно для standalone claims. To be removed after full E36 completion.
- **Limitations:** capacity confound (small ≠ base), N=500 низкая мощность, CIFAR-100 OOD для DINOv2, single seed, sanity на CKA сломан
- **Код:** PreE30_drift_pilot_dinov2/code/e30_run.py
- **Данные:** PreE30_drift_pilot_dinov2/results/results.json, embeddings.npz

### E36. Full coordinate drift on vision SSL (PLANNED)
- **Среда:** TBD (multi-seed fine-tune DINOv2-small или эквивалент)
- **Условия:** Multi-seed runs (≥5 seeds), идентичная capacity, N≥5000, правильная shuffled-pairs baseline для CKA
- **Метрика:** Procrustes R², Linear CKA, error bars
- **Цель:** Закрыть confound capacity и low N из PreE30; получить standalone evidence для drift на production-scale vision SSL
- **Статус:** PLANNED. Дизайн: вариант A (fine-tune ImageNet-100, 5 seeds, ~5×3 H100-часов на RunPod) или вариант B (опубликованные multi-seed runs)
- **Замещает:** PreE30 после завершения

---

## Step 1 PCA Diagnostic (Ядро Phase 2)

### E33. Step 1 PCA: last-token confound и pole stability на 5 LLM
- **Среда:** Активации residual stream на позиции последнего токена; 5 LLM (Qwen2.5-3B 36 слоёв, Gemma2-2B 26, OLMo-1B 16, Falcon-1B 22, Pythia-1.4B 32); 80 промптов × 8 категорий из Step 0 yadro_phase2 (12 апреля 2026)
- **Условия:** PCA(40), residualization linear regression на one-hot last token (41 уникальный токен), LDA 5-fold stratified CV
- **Метрика:** R²(token) на топ-7 PC, LDA на 7/15 PC до и после residualization, нормированная энтропия категорий на полюсах PC, внутрикатегориальная косинусная сплочённость, Spearman/Pearson корреляция, bootstrap stability (20 ресэмплингов 70%)
- **Результат:**
  - Path 2 (по слоям): R²(token) монотонно падает с глубиной у всех 5 LLM (Δ=−0.24÷−0.32); LDA на 15 PC после residualization монотонно растёт (Δ=+0.18÷+0.25). Step 1 PCA делался на серединных слоях у всех 5 моделей (Qwen 18/36, Gemma 13/26, OLMo 8/16, Falcon 11/22, Pythia 16/32) — это согласованный выбор, не специфика Qwen
  - Срез 1: PAIRED PC на остатках = 0 у всех 5 моделей (на сыром PCA 1–4). ASYMMETRIC PC: 2–7
  - Срез 1b: распределение асимметричных полюсов — code 8, emotional 7, abstract 4, factual 3, logical 2, spatial 1, narrative 0, ethical 0
  - Срез 1c: внутрикатегориальная косинусная сплочённость на остатках финального слоя (среднее по 5 моделям): emotional 0.128, spatial 0.086, code 0.086, abstract 0.083, factual 0.015, logical 0.003, narrative 0.000, ethical −0.019
  - Срез 1d: корреляция сплочённости с числом асимметричных появлений Spearman ρ=0.83 (p=0.011), Pearson r=0.75 (p=0.032), N=8. Spatial — outlier (высокая сплочённость, 1 появление)
  - Срез 2: bootstrap-устойчивость узлов (≥70%) — только code у Falcon (1/5 моделей). У всех остальных категорий и моделей — нет
- **Параметры:** Все 5 моделей, все срезы на одних активациях из yadro_phase2; bootstrap seed=42, 20 ресэмплингов
- **Статус:** COMPLETED (диагностика, не интервенция)
- **Факты:** Ф47, Ф48, Ф49, Ф50, Ф51, Ф52, Ф53, Ф54, Ф55
- **Код:** E31_step1_pca_diagnostic/code/{path2_layers,pc_polarities,srez1_polarity,srez1b_asym,srez1c_cohesion,srez1d_corr,srez2_bootstrap}.py
- **Данные:** E31_step1_pca_diagnostic/results/{path2,pc_polarities,srez1,srez1b,srez1c,srez1d,srez2}_output.txt + PATH2_report.md
- **Следствие для Step 2:** текущий датасет исчерпан как тест семантической геометрии; нужны контролируемые промпты (одинаковая длина, единый последний токен), съём активаций с финального слоя, не серединного

---

## EB-JEPA Two Rooms (transfer prescribed approach на planning с препятствиями)

### E34. EB-JEPA Two Rooms — prescribed_2 vs free planning
- **Среда:** EB-JEPA Two Rooms (Meta FAIR, 2602.03604), goal-conditioned navigation в среде из двух комнат с вертикальной стеной и дверью. wall_x и door_y рандомизированы между trajectory (fix_wall=False)
- **Условия:**
  - prescribed: PrescribedEncoder (MLP 2→256→256→512), вход = (x_a, y_a) агента, 199K params
  - free: ImpalaEncoder (CNN), вход = 65×65 RGB пиксели, 1.43M params
- **Общее:** RNNPredictor 793K params, regularizer = VICReg + IDM + temporal similarity, 12 эпох, batch=64, 100K episodes
- **Метрика:** planning success rate (MPPI, 200 samples × 20 iter, plan_length=90, 200 steps), 20 эпизодов, epoch 11
- **Параметры:** seed=1 (single seed). 12 эпох. Default LeCun config
- **Результат:**
  - free: SR = **55%** (11/20), mean_dist = 9.78
  - prescribed: SR = **0%** (0/20), mean_dist = 41.54
  - Probe loss: prescribed 0.006 (12× лучше free 0.072)
  - Pred loss: free 0.024 (2.2× лучше prescribed 0.051)
- **Статус:** COMPLETED as single-seed observation. **НЕ закрыт фактически по протоколу программы** (нужно ≥3 seeds). Methodological gap (2D prescribed без информации о среде) closed by E35
- **Платформа:**
  - prescribed training — Colab Pro T4 GPU (~50h)
  - free training — Windows CPU, Python 3.14, PyTorch 2.11 (~167h)
  - planning eval — Colab Pro T4 GPU (~1.5h)
- **Наблюдения:** Н1, Н2, Н3, Н4 (EVIDENCE.md)
- **Код:** E32_eb_jepa_planning/code/{eb_jepa_v3.ipynb, run_experiment_v3_windows.py, eb_jepa_planning_eval.ipynb}
- **Данные:**
  - E32_eb_jepa_planning/results/{free,prescribed}/training_results.json (12 epochs each)
  - E32_eb_jepa_planning/results/{free,prescribed}/planning_eval_results.json
  - E32_eb_jepa_planning/results/{free,prescribed}/encoder_stats.json (12 epochs aggregate stats — first 20 of 512 dim)
  - latest.pth.tar checkpoints для обоих encoder'ов (на Drive, в архивах eb_jepa_free.rar и eb_jepa_prescribed-*.zip)
  - plan_ep0/ — visualisations 6 ранних planning эпизодов на free@ep0 (1 success, 5 fail)
- **Дополнительный анализ (30.04.2026, post-hoc на encoder_stats.json):** mean shift в latent space — у free относительно нормы больше чем у prescribed (rel shift 0.69–0.78 vs 0.50–0.63). Caveat: агрегаты по 20 of 512 dim после LayerNorm, не per-sample drift
- **Следствие для E35:** prescribed нужно тестировать с координатами среды (wall_x, door_y), не только агента

### E35. EB-JEPA Two Rooms — prescribed_4 (с координатами стены и двери)
- **Среда:** EB-JEPA Two Rooms, тот же setup что E34
- **Условие:** prescribed_4 = (x_a, y_a, wall_x, door_y), z-score нормализация (mean=[31.59, 32.06], std=[16.10, 16.14] — те же что Normalizer.normalize_location в LeCun коде)
- **Encoder:** PrescribedEncoder (MLP 4→256→256→512). Только input dim изменён vs prescribed_2. Total ~199.5K params (comparable c prescribed_2 199K)
- **Probe head:** остаётся 2D (x_a, y_a) для сопоставимости с prescribed_2 и free
- **Параметры:** seed=1, 12 epochs, batch=64 — те же что E34 для прямого сравнения
- **Метрика:** planning success rate (та же что E34), 20 эпизодов, epoch 11
- **Цель:** проверить Г22 (координатная полнота как условие prescribed advantage)
- **Falsifier:** SR ≈ 0% — гипотеза о completeness опровергнута, проблема глубже (architecture mismatch, недостаточный capacity, fundamental limitation)
- **Compute:** Windows CPU, ~60-100h в фоне с auto-resume
- **Сохранение:** dual save — D:\experiments\E33_prescribed_4\results\ (local source of truth) + Drive backup (best-effort, mirrors only at end of epoch)
- **Зависимости:** нет
- **Status:** READY_TO_START
- **Follow-up:**
  - При SR > 30%: prescribed_3 = (x_a, y_a, wall_x) ablation — что важнее, стена или дверь
  - При SR ≈ 0%: hybrid run (HybridEncoder уже в коде); min-max нормализация контроль на Г14
  - При SR между 5–30%: ещё один seed для variance estimation
- **Код:** run_experiment_v4_windows.py (в работе, в /home/claude/E35/)
- **Связанные документы:** work_plan_2026_04_30.md (план программы)

### E37. CARLA prescribed safety axes — physical axes в driving (DEFERRED)
- **Среда:** CARLA synthetic, 500 clips × 100 frames @ 10fps, 256×256
- **Условия:** C1 free / C2 prescribed (4 axes: TTC, closing_v, lateral_offset, braking_margin) / C3 prescribed_frozen
- **Backbone:** V-JEPA 2.1 ViT-L (300M, frozen)
- **Метрики:** AP@(0.5s,1s,2s) lead-time, R²(z_i, GT_i) per axis per epoch, eigenvalue spectrum stability
- **Compute:** ~6h CARLA generation (Windows CPU) + ~45h Colab GPU (3 conditions × 3 seeds × 5h)
- **Status:** DEFERRED до завершения E35
- **Зависимости:**
  - E35 завершён (понятно ли prescribed approach жив на Two Rooms)
  - Стабильный Colab GPU (сейчас отпадает на CPU)
- **Готовность кода:** 3 скрипта в архиве (generate_carla_data.py, train.py, analyze.py), не тестировались
- **Имя:** **E37, не E16** (E16 уже занят double pendulum)

---
---

## Сводная таблица

| ID | Название | Среда | Данные | Seeds | Epochs | Episodes | Ключевой результат |
|---|---|---|---|---|---|---|---|
| E01 | Speech JEPA | LibriSpeech | — | pilot | — | — | +18–20pp entropy |
| E02 | Shov-JEPA Vision | Rico UI | — | 1 | — | 398 | +5% accuracy |
| E03 | LeWM State | Push-T gym | gym | 3 | 50 | 200 | 38× |
| E04 | LeWM Pixel | Push-T pixel | gym | — | 50 | — | 14.8× |
| E05 | Controls (Paper 1) | Push-T | gym | 3 | 50 | 200 | random≈prescribed |
| E06 | Cov + Drift | Push-T gym | gym | 3 | 30 | 200 | rank 2.99 → 233× worse |
| E07 | Freeze test | Push-T gym | gym | 3 | 30 | 200 | freeze@1 +20% |
| E08 | Random fixed | Push-T syn | syn | 3 | 30 | 200 | 17× stability |
| E09 | Aligned-drifting | Push-T syn | syn | 3 | 30 | 200 | aligned≈free |
| E10 | LR sweep + EMA | Push-T syn | syn | 1 | 50 | — | 4.3–7.0× all LR |
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
| E22 | Optimizer freeze | Push-T syn | syn | 3 | 30 | 200 | confound absent |
| E23 | Random 3D vs 5D | Push-T syn | syn | 3 | 30 | 200 | random_3d≈prescribed |
| E24 | Baseline 3D | Push-T syn | syn | 3 | 30 | 200 | 169× |
| E25 | 5D latent | Push-T syn | syn | 3 | 30 | 200 | 66×, random≈prescribed |
| E26 | 16D latent | Push-T syn | syn | 3 | 30 | 200 | 50×, align emerges 1.5× |
| E27 | Drift correlation | Push-T gym | gym | 3 | 30 | 200 | Pearson=0.95 |
| E28 | Dim sweep full | Push-T syn | syn | 3 | 30 | 200 | NO crossover, prescribed wins 1–11 |
| E29 | Noise control | Push-T syn | syn | 3 | 30 | 200 | drift≠noise, drift≠shift, free=222× |
| E30 | Critical window | Push-T gym | gym | 3 | 30 | 200 | 136× cliff: ~99% ущерба в 1-й эпохе |
| E31 | Sub-epoch freeze | Push-T syn | syn | 5 | 20 | 100 | SLOPE не порог (linear 2.2× best-step) |
| E32 | Sub-epoch freeze real | Push-T gym | gym | 5 | 4* | 50* | SLOPE, R²=0.977, 5/5 моно, Ф46 solid |
| PreE30 | Drift pilot DINOv2 | CIFAR-100 | — | 1 | — | — | R²=0.65, CKA=0.77 (pilot) |
| E36 | Drift full DINOv2 | TBD | TBD | ≥5 | TBD | TBD | DEFERRED |
| E33 | Step 1 PCA diagnostic | LLM activations | yadro_phase2 | — | — | 80 prompts | last-token confound на 5 LLM, узлы неустойчивы |
| E34 | EB-JEPA Two Rooms prescribed_2 vs free | EB-JEPA Two Rooms | LeCun config 100K | 1 | 12 | 100K | free SR=55%, prescribed SR=0% (single seed observation) |
| E35 | EB-JEPA Two Rooms prescribed_4 | EB-JEPA Two Rooms | LeCun config 100K | 1 | 12 | 100K | PLANNED — closing E34 gap, проверка Г22 |
| E37 | CARLA prescribed safety axes | CARLA synthetic | 500 clips | 3 | 30 | — | DEFERRED |


---

## Файлы данных

| Файл | Эксперименты | Размер | Источник |
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
| e32_slope.png | E32 | — | E32 per-seed кривые |
| seed_*.json | E32 | — | E32 per-seed raw (регенерируются run_seed.py) |

---

## Скрипты

| Файл | Эксперименты | Источник |
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

## Противоречия между экспериментами

**П1. E25 (5D prescribed works, 66×) vs E15/E16 (маятники prescribed не работает)**
- Push-T 5D prescribed нормализует все координаты → работает
- Маятники: prescribed и free получают идентичный вход → не работает
- Возможно: нормализация, различие в динамике, или наличие "лишних" (agent) координат в Push-T

**П2. ~~E13 (dim sweep: crossover at dim=4) vs E25 (prescribed_5d wins 66×)~~ CLOSED**
- E28 (full parameters) confirms: NO crossover. Prescribed wins dim 1–11.
- E13 was underpowered (100 ep, 20 epochs, 2 seeds, predictor hidden=128).
- dim=5 in E28 matches E25 exactly (66.3× vs 66.2×).

**П3. E08 (random_fixed_5d = 17× vs free) vs E23 (random_fixed_5d explodes)**
- E08: random_fixed проецирует 5→3, с bias, конкретные seeds
- E23: random_fixed проецирует 5→3, без нормализации входа, другие seeds
- Различие в реализации → разный результат. Original E08 result unreliable.


---

## Таблица соответствия номеров (слияние 20.08.2026)

| Апрельский | Новый | Что это | Статус |
|---|---|---|---|
| E30 | **E36** | Full coordinate drift on vision SSL | PLANNED |
| E31 | **E33** | Step 1 PCA, last-token confound, 5 LLM | COMPLETED (Ф47–Ф55, Г19–Г24) |
| E32 | **E34** | EB-JEPA Two Rooms prescribed_2 vs free | COMPLETED as single-seed observation (Н1–Н4) |
| E33 | **E35** | EB-JEPA Two Rooms prescribed_4 | READY_TO_START (Г25) |
| E34 | **E37** | CARLA prescribed safety axes | DEFERRED |

Июльские E30, E31, E32 и Г16, Г17, Г18 номеров не меняли.
