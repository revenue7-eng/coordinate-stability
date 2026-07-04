# Prescribed Axes: Факты и гипотезы

Дата начала: апрель 2026.
Протокол: факты — только экспериментально проверенные. Гипотезы — помечены статусом.

---

## ФАКТЫ

### Среда: Push-T (3 степени свободы: x_block, y_block, θ_block)

**Ф1. Prescribed (3D) vs Free (5D→3D): prescribed лучше в 38×**
- Prescribed: 0.004, Free: 0.157 (val loss)
- 3 сида, 50 эпох, 200 эпизодов
- Статья 1, Experiment 3 (LeWM State)

**Ф2. Prescribed (3D, 20K params) vs Free CNN (96×96 pixels, 744K params): prescribed лучше в 14.8×**
- CNN плато на эпохе 7 из 50
- Статья 1, Experiment 4 (LeWM Pixel)

**Ф3. Speech JEPA: prescribed (frozen cluster anchors) лучше free на +18–20pp entropy**
- 2×2 factorial: {GMM, k-means} × {soft, hard}
- Все prescribed условия лучше free
- Доминирующий фактор: frozen structure, не метод кластеризации
- Пилотное исследование, метрика — entropy (codebook utilization)
- Статья 1, Experiment 1

**Ф4. Shov-JEPA (Vision): 3 prescribed оси лучше 64 free: 72.5% vs 67.5%**
- Rico dataset, 398 UI screenshots
- Пилотное (398 samples, single seed, +5%)
- Статья 1, Experiment 2

**Ф5. Random fixed axes (3D) ≈ prescribed**
- Random fixed: 0.61× от prescribed (чуть лучше) при 200 ep
- Random fixed: 1.00× при 500 ep
- Free 3D same input: 4.47× хуже prescribed при 200 ep
- Фиксация важнее семантики осей
- Источник 0.61×: random_axes_control, 200 ep, 30 epochs, 3 seeds, no SIGReg (Ф39)
- Источник 1.00×: random_axes_control, 500 ep, 50 epochs, 9 runs (Ф39)
- Статья 1, Section 4.6 + random_axes_control/RESULTS.md

**Ф6. Equal-input control: free с тем же входом (x,y,θ) хуже prescribed в 7.6×**
- Prescribed: 0.000472, Free 3D same input: 0.003570
- Преимущество — от фиксации, не от доступа к информации
- Статья 1, Reviewer response

**Ф7. SIGReg на prescribed: эффект 0.6%. На free: эффект 1.9×**
- Removing SIGReg *улучшает* free encoder (0.006 vs 0.012)
- SIGReg лечит то, что prescribed предотвращает
- Статья 1, Reviewer response

**Ф8. SIGReg может вредить free encoder**
- Free без SIGReg: 0.037, Free с SIGReg: 0.156 (4.2× хуже)
- SIGReg форсирует изотропию, задача анизотропна
- Eigenvalues prescribed: [0.098, 0.078, 0.064] — отражает структуру задачи
- Eigenvalues free+SIGReg: [0.98, 0.93, 0.85] — искусственная изотропия
- Статья 2, Section 6.2

**Ф9. Free encoder имеет full rank (2.99/3) и isotropy (0.86) — и проигрывает prescribed в 233×**
- Prescribed: rank 2.91, isotropy 0.66
- Rank collapse — не причина деградации
- Статья 2, Section 3

**Ф10. Free encoder дрейфует: R² transfer < −62 после одной эпохи**
- Seed 42: −16.9, Seed 123: −62.2, Seed 777: −25.4
- Линейный декодер эпохи t катастрофически неверен на эпохе t+1
- 80% дрейфа — структурный (после Procrustes alignment)
- К эпохе 2→3: R² восстанавливается до 0.73–0.76
- Статья 2, Section 4.2

**Ф11. Freeze@1 улучшает free на 20%**
- Free unfrozen: 0.081, Freeze@1: 0.065
- Freeze@2: +2.9%, Freeze@10: −1.1% (нейтрально)
- Каузальное свидетельство: стабилизация encoder помогает
- Но freeze@1 (0.065) всё ещё 25× хуже prescribed (0.0025)
- Stability alone не sufficient
- Статья 2, Section 5.1

**Ф12. Random fixed encoder лучше free в 17×**
- Random fixed: 0.000476, Free: 0.008282
- Random fixed = frozen random orthogonal projection, zero semantic content
- Стабильность без alignment даёт порядковое преимущество
- Статья 2, Section 5.4

**Ф13. Prescribed лучше random fixed в 13×**
- Prescribed: 0.000036, Random fixed: 0.000476
- Alignment даёт дополнительное преимущество поверх stability
- Статья 2, Section 5.4

**Ф14. Rotated prescribed ≈ prescribed (1.09×)**
- Интерпретируемость осей не имеет значения
- Важно: фиксированность + правильное подпространство
- Статья 2, Section 5.4

**Ф15. Aligned-but-drifting ≈ free (или хуже)**
- Aligned-drifting linear: 0.012849 (хуже free 0.008282 в 1.55×)
- Aligned-drifting MLP: 0.008380 (≈ free)
- Encoder инициализирован на идеальных координатах → позволен дрейф → преимущество полностью потеряно
- Alignment без stability бесполезен
- Статья 2, Section 5.5

**Ф16. Stability × Alignment — не независимы, а иерархичны**
- 2×2 factorial:
  - Stable+Aligned (prescribed): 0.000039
  - Stable+Unaligned (random fixed): 0.000473
  - Unstable+Aligned (aligned-drifting): 0.012849
  - Unstable+Unaligned (free): 0.008282
- Stability effect среди aligned: 330×
- Stability effect среди unaligned: 17.5×
- Разница в 19× → сильное взаимодействие
- Stability — prerequisite. Alignment добавляет 12× только при наличии stability
- Статья 2, Section 5.5

**Ф17. Prescribed 11D хуже prescribed 3D в 20×, но лучше free 11D в 42×**
- Prescribed_3: 0.000036, Prescribed_11: 0.000732 (20× хуже prescribed_3)
- Free_11: 0.030509 (42× хуже prescribed_11)
- Оригинальные данные (E12, другой setup): free_11 (0.000060) лучше prescribed_11 (0.000381) в 6×
- Расхождение: E12 использовал фиксированный predictor hidden=128; E28 использует max(128, dim*8)
- С правильным predictor capacity prescribed_11 побеждает free_11
- Избыточные prescribed оси ухудшают prescribed (20× vs prescribed_3), но не настолько чтобы free победил
- Верифицировано: p2_dim_sweep_results.json (E28)
- 200 эпизодов, 30 эпох, 3 сида

**Ф18. ~~Sweep dim 1→15: prescribed побеждает только при dim ≤ 3~~ ОПРОВЕРГНУТ**
- Оригинальные данные (100 ep, 20 epochs, 2 seeds) показывали crossover при dim=4
- **Полный перезапуск (200 ep, 30 epochs, 3 seeds) опроверг crossover:**
- dim=1: 60×, dim=2: 1820×, dim=3: 228×, dim=4: 114×, dim=5: 66×, dim=7: 57×, dim=11: 42×
- Prescribed побеждает на ВСЕХ размерностях 1–11
- Gap монотонно убывает с ростом dim (228× → 42×), но не исчезает
- dim=2 аномально высокий (1820×) — prescribed [x_b, y_b] идеально matched
- dim=5 совпадает с Tier 3 E25 (66.3× vs 66.2× — полная воспроизводимость)
- Оригинальный результат был артефактом underpowered setup
- Верифицировано: p2_dim_sweep_results.json, per-seed breakdown
- Эксперимент E28 (15.04.2026)

**Ф19. Prescribed не выигрывает ни при какой dim (1–5)**
- Free лучше при всех размерностях
- Prescribed и free получают одинаковый 2D вход
- 100 эпизодов, 20 эпох, 3 сида
- Эксперимент 15.04.2026

**Ф21. Непредсказуемая ось убивает prescribed катастрофически (1106×)**
- prescribed_4_noise (x,y,θ + frozen random): 0.021534
- prescribed_3 (x,y,θ): 0.000019
- Предиктор обязан предсказать непредсказуемое → полный провал
- 200 эпизодов, 30 эпох, 3 сида
- Эксперимент 15.04.2026 (fragility test)

**Ф22. Оси из того же подпространства вредят меньше, чем из другого**
- prescribed_4_sin (x,y,θ + sinθ): 0.000093 (4.8× vs p3) — redundant, same subspace
- prescribed_4_dist (x,y,θ + d_agent_block): 0.000160 (8.2× vs p3) — redundant, cross-subspace
- prescribed_4_agent (x,y,θ + agent_x): 0.000154 (7.9× vs p3) — independent, cross-subspace
- sin(θ) ≈ вдвое лучше чем dist и agent_x
- Подпространство имеет значение: ось из того же субпространства (блок) менее вредна
- 200 эпизодов, 30 эпох, 3 сида
- Эксперимент 15.04.2026 (fragility test)

**Ф23. Избыточность vs независимость 4-й оси: эффект одинаковый при cross-subspace**
- prescribed_4_dist (redundant, cross-subspace): 0.000160 (8.2×)
- prescribed_4_agent (independent, cross-subspace): 0.000154 (7.9×)
- Разница менее 5% → дело не в избыточности как таковой, а в расширении подпространства
- 200 эпизодов, 30 эпох, 3 сида
- Эксперимент 15.04.2026 (fragility test)

### Среда: Двойной маятник (4 степени свободы: θ1, ω1, θ2, ω2)

**Ф20. ~~Prescribed выигрывает только при dim=1~~ ОПРОВЕРГНУТ — нормализация решает**
- Оригинал (без нормализации): prescribed проигрывает free при dim ≥ 2
- **С нормализацией [0,1]: prescribed побеждает free на ВСЕХ dim 1–8:**
  - dim=1: 9.8×, dim=2: 16.8×, dim=3: 12.3×, dim=4: 19.1×
- Нормализация даёт 37–166× улучшение prescribed encoder
- Prescribed raw: 0.05–0.21, prescribed norm: 0.001–0.003, free: 0.014–0.035
- Prescribed и free получают одинаковый 4D вход — нет отбора
- **П1 ЗАКРЫТ: различие Push-T vs маятники было в нормализации**
- 200 эпизодов, 30 эпох, 3 сида, per-seed verified
- Эксперимент E16 fix (15.04.2026)
- Данные: fix_e16_results.json

### Tier 1 тесты (критические тесты гипотез, 15.04.2026)

**Ф24. Ранний drift (ep 0→1) уничтожает информацию — даже MLP decoder ломается**
- Epoch 0→1: MLP decoder R² transfer = −283 (среднее по 3 seeds)
- Linear decoder R² transfer = −71 (среднее по 3 seeds)
- MLP self R²: 0.80–0.95 (хорошо обучен на epoch t)
- MLP decoder НЕ может восстановить GT из embeddings epoch t+1
- Информация реально уничтожена, не просто линейно нечитаема
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 1 / T1

**Ф25. Поздний drift (ep 2+) сохраняет информацию в нелинейно изменённой форме**
- Epoch 5→6: MLP decoder R² transfer = 0.79, linear = 0.69
- Epoch 29→30: MLP decoder R² transfer = 0.81, linear = 0.69
- MLP advantage (late epochs): +0.12 R² в среднем
- Информация есть, но линейно нечитаема — MLP восстанавливает
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 1 / T1

**Ф26. Differential LR (encoder 100× медленнее) сокращает gap на 72%, но 62× gap остаётся**
- Prescribed: 0.000037
- Free K=1: 0.008282 (222×)
- Free diffLR 100× (enc LR=3e-6): 0.002303 (62×)
- Extra predictor steps (K=3): 0.010399 — ХУЖЕ чем K=1
- Extra predictor steps (K=5): 0.006377 (171×)
- DiffLR 10× (enc LR=3e-5): 0.007612 (204×) — слабый эффект
- Проблема НЕ чисто optimization lag
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 1 / T2

**Ф27. PCA canonicalization не помогает — drift нелинейный**
- PCA ухудшает R² transfer на большинстве эпох
- Epoch 0→1: raw = −71.4, PCA = −69.1 (минимальная разница)
- Epoch 2→3: raw = 0.678, PCA = −0.376 (PCA ХУЖЕ)
- Epoch 29→30: raw = 0.690, PCA = 0.298 (PCA ХУЖЕ)
- Drift — нелинейная деформация, не rotation/scaling
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 1 / T3

**Ф28. Drift rate коррелирует с val loss: Pearson = 0.95**
- Корреляция доминируется epoch 0→1 (outlier)
- Spearman = 0.51 (слабее — зависимость нелинейная)
- Два режима: "зона катастрофы" (drift > 0.3) и "зона насыщения" (drift < 0.1)
- После эпохи 3 drift перестаёт быть bottleneck, доминирует alignment
- R² потолок free encoder ≈ 0.75 (linear), prescribed ≈ 1.0
- Данные из all_results.json (gym-pusht), 3 сида, 30 эпох
- Tier 1 / T8

### Tier 2 тесты (confound tests, 15.04.2026)

**Ф29. SIGReg стабилизирует aligned-drifting linear (не уничтожает)**
- Aligned-linear с SIGReg: 0.013314 (стабильно по seeds: 0.013, 0.016, 0.011)
- Aligned-linear без SIGReg: 0.470604 (нестабильно: 0.016, 1.393, 0.003)
- Seed 123 без SIGReg: 1.39 — catastrophic divergence
- SIGReg предотвращает divergence линейного encoder
- Но ни с SIGReg (356×), ни без — aligned-drifting не приближается к prescribed
- Free с SIGReg: 0.008282, Free без SIGReg: 0.007345 (SIGReg слегка вредит free — подтверждает Ф8)
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 2 / T4

**Ф30. Optimizer reset в freeze test — НЕ confound**
- freeze@1 new optimizer: 0.008785 (−6.1% vs unfrozen)
- freeze@1 keep state: 0.008701 (−5.1% vs unfrozen)
- Разница: 1.0%
- freeze@3 new optimizer: 0.007108 (+14.2%)
- freeze@3 keep state: 0.006898 (+16.7%)
- Разница: 2.9%
- Optimizer state preservation не меняет результат freeze test
- НО: freeze@1 на synthetic данных ухудшает (−6%), а не улучшает (+20% на gym-pusht)
- Freeze эффект зависит от данных, но optimizer confound отсутствует
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 2 / T5

**Ф31. Random fixed 3D (из блочных координат) ≈ prescribed = rotated prescribed**
- prescribed: 0.000037
- rotated_prescribed: 0.000039 (1.05×)
- random_fixed_3d: 0.000036 (0.97×)
- Все три ≈ одинаковы
- Любой стабильный ортогональный базис в правильном подпространстве = prescribed
- Alignment осей внутри подпространства не имеет значения
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 2 / T7

**Ф32. Random fixed 5D (из всех координат, без нормализации) — взрывается**
- random_fixed_5d: 376,053 (среднее; per seed: 977K, 25K, 125K)
- Ненормализованная проекция из полного пространства нестабильна
- Оригинальный "17× stability advantage" из Paper 2 (Ф12) был получен на конкретных seeds с другой реализацией random_fixed
- Tier 2 / T7

### Tier 3 тесты (generalization, 15.04.2026)

**Ф33. Prescribed vs Free gap воспроизводится в 3D, 5D и 16D**
- 3D: prescribed 0.000050, free 0.008503, gap 169×
- 5D: prescribed 0.000354, free 0.023431, gap 66×
- 16D: prescribed 0.000747, free 0.037105, gap 50×
- Gap уменьшается с ростом размерности, но остаётся порядковым
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 3 / T9a, T9b

**Ф34. Random fixed ≈ prescribed в 5D (0.92×), начинает отставать в 16D (1.53×)**
- 5D: random_fixed = 0.000324, prescribed = 0.000354 (random чуть лучше!)
- 16D: random_fixed = 0.001142, prescribed = 0.000747 (prescribed лучше в 1.53×)
- В 5D (линейные координаты): alignment внутри подпространства не важен
- В 16D (нелинейные features): alignment начинает иметь значение
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 3 / T9a, T9b

**Ф35. Drift усиливается с размерностью**
- 3D: drift_01 = 1.53, R² transfer = −70
- 5D: drift_01 = 1.91, R² transfer = −65
- 16D: drift_01 = 3.58, R² transfer = −596
- R² transfer в 16D в 8.5× хуже чем в 3D
- Больше размерность → больше "плоских направлений" → сильнее drift
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 3 / T9a, T9b

**Ф36. 5D prescribed (все координаты, без отбора) работает — gap 66×**
- Prescribed_5d = normalize(все 5 координат), free_5d = MLP 5→5
- Prescribed НЕ выбирает подпространство — берёт всё
- Gap 66× — меньше чем 3D (169×), но всё ещё порядковый
- Contradict Ф19 (маятники): там prescribed на полном входе не работает
- Разница: Push-T 5D prescribed нормализует, маятники — нет (?)
- Или: Push-T содержит "лишние" координаты (agent), маятники — нет
- 200 эпизодов, 30 эпох, 3 сида, synthetic
- Tier 3 / T9a

### Верифицированные из архива Paper 1 (15.04.2026)

**Ф37. Gauge fixing free encoder не помогает (1.08× ≈ free)**
- prescribed: 0.000472, free: 0.011614, gauge_fixed_free: 0.012581
- linear_free: 16009 (взрыв)
- Gauge fixing (фиксация симметрии через training) не решает drift
- Data seed 42, training seeds [42, 123, 777], 50 epochs, synthetic
- Дата: 12.04.2026
- Источник: 1.rar/gauge_fix_results/results.json
- Не был включён в FACTS.md ранее

**Ф38. При 500 эпизодах free encoder побеждает prescribed в 695,000×**
- prescribed: 8.513×10⁻⁴ (3 seeds, std 8.4×10⁻⁷)
- random_fixed: 8.514×10⁻⁴ (9 runs: 3 rotation seeds × 3 training seeds, std 2.1×10⁻⁶)
- free_3d: 1.225×10⁻⁹ (3 seeds, std 3.7×10⁻¹⁰)
- free_5d: 2.916×10⁻⁹ (3 seeds, std 7.8×10⁻¹⁰)
- Fixed encoders (prescribed и random) плато на ~8.5×10⁻⁴ — irreducible error от min-max нормализации
- Free encoder сходится к ~10⁻⁹ (фактически нулю)
- **Prescribed axes = sample efficiency, не абсолютное преимущество**
- 500 эпизодов, 50 эпох, synthetic, no SIGReg
- Верифицировано из JSON: exp5_random_axes/all_results.json (18 runs)
- Paper 1, random_axes_control/RESULTS.md

**Ф39. Random_fixed ≈ prescribed при обоих масштабах данных**
- 200 ep: random 0.61× (чуть лучше prescribed) — 3 seeds, 30 epochs
- 500 ep: random 1.00× (идентично) — 9 runs vs 3 runs
- Axis semantics вторичны при любом количестве данных
- Верифицировано из JSON: random_fixed_results/results.json (200ep), all_results_500ep.json (500ep)
- Paper 1, random_axes_control/RESULTS.md

**Ф40. Isotropic normalization (zero-mean, unit-variance) ухудшает в 15×**
- prescribed_iso: 0.010250 vs prescribed: 0.000673 (15.2×)
- random_fixed_iso: 0.007019 vs random_fixed: 0.000408 (17.2×)
- Min-max [0,1] нормализация критична; standardization вредит
- 200 эпизодов, 20 epochs, synthetic
- Paper 1, random_axes_control/RESULTS.md

### Noise control (E29, 17.04.2026)

**Ф41i. i.i.d. noise при drift-matched амплитуде (ep 0→1, σ≈0.90) разрушает prescribed в 851×**
- noise_early: 0.031781, prescribed: 0.000037
- Free encoder: 0.008282 (222×) — лучше i.i.d. noise в 4×
- Drift не сводится к random perturbation
- 3 seeds, 200 эпизодов, 30 эпох, synthetic
- E29

**Ф42i. Correlated noise (constant shift per epoch) почти не вредит prescribed: 1.3× даже при катастрофической амплитуде**
- correlated_schedule: 0.000050, prescribed: 0.000037
- Predictor компенсирует constant displacement: разности между timesteps сохраняются
- 3 seeds, 200 эпизодов, 30 эпох, synthetic
- E29

**Ф43i. Free encoder (222×) в 167× хуже correlated noise (1.3×) при matched амплитуде**
- free: 0.008282, correlated_schedule: 0.000050
- Drift ≠ global coordinate shift. Data-dependent деформация.
- 3 seeds, 200 эпизодов, 30 эпох, synthetic
- E29

**Ф44i. Спектр деградации: prescribed (1×) < correlated (1.3×) < noise_mid (6.2×) < FREE (222×) < noise_early (851×)**
- Free encoder ближе к i.i.d. noise, чем к correlated shift
- Drift вносит несогласованность внутри context window, не просто глобальный сдвиг
- 3 seeds, 200 эпизодов, 30 эпох, synthetic
- E29

### Критическое окно / sub-epoch (drift-hallucination ветка, 03.07.2026)

**Ф45. Критическое окно повреждения free-энкодера = внутри первой эпохи (E30)**
- freeze@0 (random_fixed прокси, Ф12 = 0.000476) → freeze@1 (0.06485, gym-pusht) = **136× cliff**
- freeze@1 → unfrozen (0.08113) = **1.3×** (пренебрежимо на фоне cliff)
- ~99% повреждения free-энкодера наносится в первую эпоху; обрыв между эпохой 0 и 1, не позже
- band freeze@k≥1 (k=1,2,3,5,7,10) узкий: 0.065–0.082, каждый ≥25× хуже prescribed (0.00252)
- Механистически замыкает Ф31: random_fixed ≈ prescribed, т.к. оба = freeze@0 (представление ДО катастрофической первой эпохи)
- Метод: анализ уже записанного freeze@k-профиля из E06/E07 all_results.json, без новых прогонов; воспроизводится точно (repro 136.25× = shipped)
- Среда: Push-T (gym-pusht, real pymunk, synthetic=false), 3 сида, 30 эпох, 200 эпизодов
- Оговорки: freeze@0 = прокси (Ф12), не литеральный прогон (нормализация может отличаться); 3 сида — сигнал, не доказательство; sub-epoch разрешения нет; freeze data-dependent (Ф30)
- Артефакт: E30_critical_window/ (README + code + results, воспроизводимо из all_results.json)
- E30

**Ф46. Внутри первой эпохи повреждение — непрерывный СКЛОН, не дискретный порог (E31 synthetic → E32 real, SOLID)**
- Вопрос: внутри эпохи 1 ущерб появляется на резком пороге (дискретный момент невозврата) или накапливается как склон? Энкодер замораживается на долях батчей эпохи 1.
- **E31 (synthetic, 5 сидов):** verdict SLOPE (от analyze_shape.py). Linear-vs-step над подъёмом f≥0.25: линия лучше в 2.2× (SS 0.085 vs 0.186), linear R²=0.88. Монотонно 3/5 сидов (0 спадов), 2/5 по одному шумовому спаду.
- **E32 (real gym-pusht, 5 сидов {7,42,123,777,2024}):** verdict SLOPE, ЧИЩЕ synthetic. **5/5 сидов строго монотонны** (0 спадов любого рода). Pooled linear-vs-step над band f∈[0.25,0.60] (per-seed min-max норм.): **linear R²=0.977**; лучший single-breakpoint step хуже в **10.5×** (SS_step 1.047 vs SS_lin 0.100, breakpoint f=0.50). Raw per-seed кривые слегка выпуклы (инкременты растут к f=0.60). E30-style anchor на реальных данных: freeze@1.0/@0.0 = **22.1×** (per seed 7.2–35.3×) — то же направление, что 136× cliff E30.
- Первая четверть эпохи (f≤0.25) near-harmless; далее повреждение интегрируется непрерывно, ускоряясь к концу
- Против дискретного события необратимости; за непрерывный дрейф. E32 снимает единственную (synthetic) оговорку с Ф46 → **SOLID**.
- Среда: E31 — Push-T synthetic (synth(), 20 epoch/100 ep); E32 — Push-T real gym-pusht (reduced budget EP=4/NEP=50, pymunk 6.2.1 pinned)
- Оговорки (E32): reduced budget сжимает АБСОЛЮТНЫЕ gap (prescribed/unfrozen ~4× здесь vs 222× at scale; anchor cliff 22× vs 136× у E30) — числа масштаба НЕ сравнивать с 30-эпоховыми прогонами; **shape verdict budget-robust** (кривая монотонна и slope-shaped независимо от бюджета). pymunk 6.2.1 (gym_pusht просит ≥6.6, но 6.6+/7.x ломают add_collision_handler). Full-fidelity rerun (EP=15,NEP=200) = one-line change.
- [ЗАМЕТКА для сверки] Разрешения в (0.0, 0.25) в этих артефактах НЕТ (grid {0.0, 0.25…0.60, 1.0}); near-harmless onset здесь качественный (f≤0.25). Формулировка «зона [0.00–0.20] slope≈1.25 vs [0.25–0.60] slope≈13.04, ratio 10.4×» требует отдельного higher-resolution прогона (sub-0.25) и этими файлами не подтверждена. Числа выше — из shipped shape_verdict.json.
- Артефакты: E31_subepoch_freeze/, E32_subepoch_freeze_real/ (verdict в shape_verdict.txt / shape_verdict.json)
- E31, E32

---

## ГИПОТЕЗЫ

### Подтверждённые (на одной среде)

**Г1. Фиксация осей важнее семантики осей**
- Подтверждено: random fixed ≈ prescribed (Ф5, Ф12, Ф31)
- Усилено: random_fixed_3d ≈ prescribed ≈ rotated_prescribed (Ф31) — alignment внутри подпространства не имеет значения
- Воспроизведено в 5D (Ф34: random ≈ prescribed, 0.92×)
- В 16D начинает ослабевать (Ф34: 1.53×) — при нелинейных features alignment появляется
- Среда: Push-T (3D, 5D)
- Статус: ПОДТВЕРЖДЕНА на Push-T, с оговоркой для high-dim

**Г2. Дрейф координат — причина деградации free encoder**
- Подтверждено: R² < −62 (Ф10), freeze@1 +20% (Ф11), aligned-but-drifting ≈ free (Ф15)
- Усилено (Tier 1):
  - Ранний drift уничтожает информацию — MLP decoder тоже ломается (Ф24)
  - Поздний drift сохраняет информацию нелинейно (Ф25)
  - Drift нелинейный — PCA не может исправить (Ф27)
  - diffLR 100× оставляет 62× gap — не optimization lag (Ф26)
- Freeze test confound отсутствует (Ф30)
- Drift усиливается с размерностью (Ф35)
- Среда: Push-T (3D, 5D, 16D)
- Статус: ПОДТВЕРЖДЕНА, усилена

**Г3. Rank collapse — не главная причина деградации free encoder**
- Подтверждено: full rank 2.99 + isotropy 0.86 → всё равно 233× хуже (Ф9)
- Среда: Push-T
- Статус: ПОДТВЕРЖДЕНА на Push-T

**Г4. Stability — prerequisite, alignment — дополнительный фактор**
- Подтверждено: 2×2 factorial (Ф16), aligned-but-drifting ≈ free (Ф15)
- Уточнено (Tier 2): alignment *внутри подпространства* не важен (Ф31)
- Alignment = выбор правильного подпространства + нормализация, не ориентация осей
- Среда: Push-T
- Статус: ПОДТВЕРЖДЕНА, уточнена — "alignment" переопределено как "subspace selection"

### Опровергнутые

**Г5. Crossover prescribed/free = внутренняя размерность задачи**
- Push-T: ~~crossover 3→4~~ НЕТ CROSSOVER на полных данных (E28). Prescribed wins 1–11.
- Маятник: crossover нет, внутр. dim=2 — (данные Ф19, не верифицированы с полными параметрами)
- Двойной маятник: crossover 1→2, внутр. dim=4 — (данные Ф20, перезапускается)
- Оригинальный crossover на Push-T был артефактом underpowered E13 (100 ep, 20 epochs, 2 seeds)
- Статус: ОПРОВЕРГНУТА (crossover не существует на Push-T при полных параметрах)

**Г6. 11 осей (по аналогии с M-теорией) дадут лучший результат**
- Prescribed_11 хуже prescribed_3 в 20× (Ф17)
- Статус: ОПРОВЕРГНУТА

**Г10. Prescribed advantage = чистый stability effect (17× × 13× = 233× decomposition)**
- Decomposition из Paper 2 невалидна:
  - random_fixed_5d (из Ф12) нестабилен — взрывается на новых seeds (Ф32)
  - random_fixed_3d (из правильного подпространства) ≈ prescribed (Ф31)
  - "17× stability" был артефактом конкретной реализации random_fixed
- Правильная decomposition: subspace selection + normalization + freeze
- Alignment осей внутри подпространства не фактор
- Статус: ОПРОВЕРГНУТА (decomposition, не thesis)

**Г11. Проблема free encoder — optimization lag (решается scheduler/LR)**
- diffLR 100× помогает на 72%, но 62× gap остаётся (Ф26)
- Extra predictor steps (K=3) УХУДШАЮТ результат (Ф26)
- EMA (decay=0.996) не помогает (Paper 2, 6.1×)
- Статус: ОПРОВЕРГНУТА — optimization lag частичный фактор, не главная причина

**Г12. Drift — rotation/scaling, решается PCA canonicalization**
- PCA ухудшает R² transfer на большинстве эпох (Ф27)
- Drift нелинейный
- Статус: ОПРОВЕРГНУТА

**Г13. SIGReg разрушает aligned initialization**
- SIGReg стабилизирует aligned-drifting linear, предотвращает divergence (Ф29)
- Без SIGReg: seed 123 → catastrophic divergence (1.39)
- Статус: ОПРОВЕРГНУТА — SIGReg стабилизирует, не разрушает

### Открытые

**Г7. ~~Prescribed работает не из-за фиксации, а из-за отбора информации~~ ОПРОВЕРГНУТА**
- Push-T: prescribed побеждает при dim=5 (все координаты, без отбора) в 66× (E28)
- Двойной маятник: prescribed_norm побеждает при dim=4 (все координаты, без отбора) в 19× (Ф20)
- Оригинальный аргумент (маятники не работают → нужен отбор) был артефактом отсутствия нормализации
- Статус: ОПРОВЕРГНУТА — фиксация + нормализация достаточна, отбор не нужен

**Г8. ~~Prescribed работает из-за комбинации фиксации + отбора~~ ОПРОВЕРГНУТА**
- Фиксация без отбора работает на обеих средах при нормализации:
  - Push-T dim=5 (все координаты): 66× (E28)
  - Double pendulum dim=4 (все координаты): 19× (Ф20 updated)
- Отбор не является необходимым компонентом
- Correct formulation: prescribed = фиксация + нормализация [0,1]
- Статус: ОПРОВЕРГНУТА

**Г9. Хрупкость prescribed: одна лишняя ось убивает преимущество**
- ~~Факт: dim=3→4 = потеря на Push-T (Ф18)~~ ОПРОВЕРГНУТ: prescribed wins dim=4 в 114× (E28)
- Механизм из E17 (Ф21–Ф23) остаётся: непредсказуемая ось катастрофична (1106×)
- Но: предсказуемые лишние оси ухудшают prescribed (228× → 42× при dim 3→11), не убивают
- Gap монотонно убывает, но prescribed побеждает на всех размерностях
- **E12 (free_11 > prescribed_11) был артефактом маленького predictor (hidden=128)**
- С predictor capacity max(128, dim*8) prescribed побеждает даже при dim=11
- Статус: ЧАСТИЧНО ОПРОВЕРГНУТА — хрупкость к noise осям реальна (Ф21), но лишние предсказуемые оси не убивают prescribed advantage

**Г14. Prescribed advantage = фиксация + нормализация [0,1] координат**
- На основе всех экспериментов (Tier 1-3, E28, E16 fix)
- Prescribed выигрывает когда:
  (a) фиксирует координаты (не дрейфует)
  (b) **нормализует координаты в [0,1] (min-max)** — ОБЯЗАТЕЛЬНО
  (c) координаты содержат релевантную информацию о задаче
- Без нормализации prescribed проигрывает (маятники raw: 0.1–0.4×)
- С нормализацией prescribed побеждает на всех средах и всех dim:
  - Push-T dim 1–11: 42–1820× (E28)
  - Double pendulum dim 1–8: 10–19× (Ф20 updated)
- Alignment осей внутри подпространства не имеет значения (Ф31)
- Random_3d ≈ prescribed при нормализации (Ф31, Ф34)
- Standardization (zero-mean, unit-var) ухудшает в 15× (Ф40)
- **При достаточных данных (500 ep) free encoder побеждает prescribed в 695,000× (Ф38)**
- Prescribed = sample efficiency + guaranteed stability, не абсолютное преимущество
- Gauge fixing не работает как альтернатива (Ф37)
- Статус: ПОДТВЕРЖДЕНА на Push-T и двойном маятнике (2 среды, 3+ размерности)

**Г15. Двухфазная модель drift**
- Новая гипотеза на основе Tier 1
- Фаза 1 (ep 0–2): катастрофическая — информация уничтожается нелинейно и невосстановимо
- Фаза 2 (ep 3+): стабилизация — информация сохраняется, но в нелинейно дрейфующем пространстве
- Граница фаз: drift rate ≈ 0.3 (из T8 scatter plot)
- R² потолок free encoder (phase 2) ≈ 0.75 (linear), gap с prescribed определяется alignment, не drift
- Статус: НОВАЯ, подтверждена на Push-T 3D (Ф24, Ф25, Ф28)

**Г16. Free-латент имеет характеризуемую скорость дрейфа (drift-rate law)**
- Импорт из INS/GNSS-denied (SBG Systems, триаж 07.2026): dead-reckoning моделирует дрейф как error-growth — накопленная ошибка = drift-rate × время-с-момента-изъятия-якоря. Перенос: накопленное отклонение базиса free-энкодера растёт как функция числа эпох без prescribed-ограничения.
- Частично уже наблюдается (сырьё для rate-fit существует, 3 seeds):
  - Ф10: per-epoch R² transfer (seed 42 −16.9, 123 −62.2, 777 −25.4; восстановление к ep 2→3)
  - Г15: двухфазная модель, drift rate ≈ 0.3 как граница фаз (из T8 scatter)
- Новое предсказание (тестируемо на Push-T): накопленное отклонение базиса (Procrustes к якорной истине) — монотонная функция эпох-с-изъятия-якоря с оцениваемым rate; drift-rate(free) >> drift-rate(prescribed ≈ random_fixed ≈ 0).
- Метод измерения (импорт из SBG):
  - ablation-by-withholding: якорь изъят у энкодера, сохранён как ground truth для замера (аналог Qinertia GNSS rejection module — симуляция outage без физического отключения)
  - мерить траекторию базиса по эпохам, НЕ эндпоинт: эндпоинт-метрика (напр. финальный Procrustes R²=−15.7) маскирует дрейф процесса — прямой аналог конфаунда «петли и повороты маскируют дрейф курса» и last-token confound E31
- Горизонт-конъектура (НЕ тестируется на Push-T, cross-domain — не факт): превышение порога накопленного дрейфа = переход в hallucination basin (2604.04743, статические basins). drift-rate × t → порог даёт динамику перехода. Отдельная статья drift→hallucination.
- Среда: Push-T (rate-fit возможен из существующих E-данных Ф10/Г15). Горизонт-часть — вне среды.
- **ОПРОВЕРЖЕНИЕ (03.07.2026, проверка на all_results.json, без новых прогонов):**
  - [FACT] free raw_drift падает 1.43 (ep1) → 0.001 (ep30); отношение поздний/ранний = 0.002 — дрейф выгорает за ~3 эпохи.
  - [FACT] Кумулятивный fit: linear (постоянный rate) R²=0.821 < sqrt (диффузия) 0.920 < log (насыщение) 0.977. Постоянного rate НЕТ — дрейф это затухающий front-loaded transient, не накопление.
  - [INFERENCE] Аналогия INS/GNSS dead-reckoning ложна: у INS ошибка растёт неограниченно во времени, здесь — самозатухает. «drift-rate × t → порог» строить не на чем.
  - [INFERENCE] Данные подтверждают Ф24/Ф25 (двухфазная модель), НЕ Г16.
  - Оговорка: «log R²=0.977» ≠ «логарифмический закон» — артефакт front-loading; prescribed=0 вырожден.
  - Артефакт: E30_critical_window/finding_drift_rate.docx
- Статус: **ОПРОВЕРГНУТА** — rate-форма (постоянный/характеризуемый rate) не подтверждена данными; дрейф — front-loaded затухающий transient (фаза 1 Ф24/Ф25). Горизонт-конъектура drift→hallucination лишается кинематической опоры в этой форме.

**Г17. Epiplexity ⊥ identifiability (structural information слепа к дрейфу координат)**
- Тезис: epiplexity (S_T, структурная информация в смысле time-bounded MDL, Finzi et al. arXiv:2601.03220) мерит *количество* поглощённой структуры, оставаясь агностичной к её заземлённости/идентифицируемости. Модель может поглощать большой объём структуры (низкий NLL → высокая S_T) при сломанной идентифицируемости координат (R² < 0 после Procrustes). → epiplexity и стабильность координат — ортогональные оси.
- Опора (их собственные свойства):
  - epiplexity content-agnostic по определению (Finzi §6.1): «measures the amount of structural information, irrespective of its content».
  - epiplexity loss-derived (prequential = площадь под loss-кривой выше финала). Видит объём сжатия, не геометрию базиса. → сильнейшая инстанциация тезиса «пространство важнее лосса»: даже информационно-теоретическое прочтение loss-кривой не различает заземлённую структуру и внутренне согласованный дрейфующий код.
- Наблюдаемое на данный момент — только low-id: free R²=−15.7 (Procrustes, финальный) при полном rank/isotropy (Ф9) и приличном NLL. Это Н (1 наблюдение).
- НЕ наблюдаемое (пока): epiplexity free/prescribed энкодеров не измерена. Ячейка «high-epi + low-id» — ПРЕДСКАЗАНИЕ, не факт.
- Confound (потенциальный вклад): prequential-оценщик предполагает сходимость к стабильному финальному лоссу. Под дрейфом «финал» плохо определён → S_preq может быть раздута несходимостью, смешивая «поглощённую структуру» с «неспособностью заземлиться». В статье Finzi это допущение не проверяется. Прямой аналог конфаунда эндпоинт-vs-траектория (Г16) и last-token confound (E31).
- Домены (не конфлатить): JEPA — epiplexity лишь прокси (нет явного правдоподобия), сюда её НЕ тащить как метрику. Чистый дом — LLM-латент (likelihood-модель), т.е. тред drift→hallucination, не prescribed-axes статьи.
- Отличие от Г16 (называю явно, чтобы не склеить): Г16 = дрейф характеризуем как rate во времени (кинематика, dead-reckoning). Г17 = метрика объёма структуры не видит дрейф вообще. Разные claims; общий только горизонт hallucination. Не объединять.
- Проверка (likelihood-домен): на LLM-чекпойнтах S_preq (из существующих NLL-логов) × метрика дрейфа/идентифицируемости латента. Подтверждение = ячейка high-epi/low-id непуста, S_preq не различает заземлённую и дрейфующую структуру при разбиении по идентифицируемости. Опровержение = S_preq монотонно коррелирует с идентифицируемостью (тогда epiplexity — симптом, не ортогональная ось).
- Среда: вне Push-T (домен — LLM). Для JEPA неприменима как метрика.
- Статус: ОТКРЫТА / ГИПОТЕЗА. Опора = 1 Н (low-id) + непроверенное предсказание (epiplexity не измерена). НЕ факт: нужен LLM-тест на ≥3 seed с явным измерением S_preq. Связи: Paper 1 «Space Matters More Than the Loss»; Г16; hallucination basins (2604.04743); отдельная статья drift→hallucination после фазы 2 Ядро.

**Г18. Критическое окно повреждения free-энкодера = внутри первой эпохи; форма — непрерывный склон, не дискретный порог**
- Уточняет Г15 (двухфазная модель): фаза 1 — не «эпохи 0–2», а «внутри первой эпохи».
- Опора: Ф45 (E30) — ~99% повреждения в первой эпохе (freeze@0→@1 = 136× cliff vs freeze@1→unfrozen = 1.3×); Ф46 (E31 synthetic + E32 real) — внутри первой эпохи это склон (SLOPE verdict от кода: E32 R²=0.977, step хуже 10.5×, 5/5 монотонны).
- Первая четверть эпохи (f≤0.25) near-harmless; далее непрерывная интеграция ущерба, ускоряясь к концу.
- **Корректная формулировка (важно для галлюцинационного моста):** «early, continuously-integrated divergence that later training does not undo» — НЕ «irreversible event». Слово «необратимое» не должно читаться как дискретность: необратимость — свойство терминального состояния эпохи 1 относительно позднего обучения (E30/Ф45), достигаемое непрерывным накоплением (E31/E32/Ф46). Склон прямо опровергает дискретный порог.
- Мост к галлюцинациям: общая ось — уверенный выход при незаземлённом состоянии (НЕ механизм-тождество). Настоящий тест домена — LLM (Г17, epiplexity ⊥ identifiability), вне Push-T. Держать на поводке.
- Среда: Push-T (synthetic E31 + real gym-pusht E32). Домен галлюцинаций — отдельно (Г17).
- Статус: ПОДТВЕРЖДЕНА на Push-T (E30 real + E31 synthetic + E32 real). Оговорка E32: reduced budget → абсолютные магнитуды сжаты, shape verdict budget-robust.

---

## КЛЮЧЕВЫЕ РАЗЛИЧИЯ МЕЖДУ СРЕДАМИ

| Свойство | Push-T | Двойной маятник (raw) | Двойной маятник (norm) |
|---|---|---|---|
| Внутр. размерность | 3 | 4 | 4 |
| Полный state dim | 5 | 4 | 4 |
| Prescribed wins? | Да (dim 1–11) | Нет (Ф20 original) | **Да (dim 1–8)** |
| Нормализация | [0,1] min-max | Нет | [0,1] min-max |
| Gap prescribed/free | 42–1820× | 0.1–0.4× | **10–19×** |

---

## КЛЮЧЕВЫЕ ПРОТИВОРЕЧИЯ (требуют разрешения)

**П1. ~~Push-T 5D prescribed работает (Ф36), но маятники prescribed на полном входе — нет (Ф19, Ф20)~~ ЗАКРЫТ**
- Причина: отсутствие нормализации в маятниках
- С нормализацией [0,1] prescribed побеждает free на двойном маятнике при ВСЕХ dim 1–8 (Ф20 updated)
- Нормализация даёт 37–166× улучшение prescribed
- Push-T prescribed всегда нормализовал — поэтому работал
- Маятники без нормализации — raw координаты в разных масштабах (θ ∈ [-π,π], ω ∈ [-10,10])
- Ф19 (simple pendulum) — предположительно тот же артефакт, нужна верификация

**П2. ~~Dim sweep (Ф18): prescribed проигрывает при dim≥4. Но Ф36: prescribed_5d выигрывает в 66×~~ ЗАКРЫТ**
- E13 (100 ep, 20 epochs, 2 seeds) был underpowered
- E28 (200 ep, 30 epochs, 3 seeds, predictor max(128, dim*8)): prescribed wins 1–11, no crossover
- dim=5 совпадает с Tier 3 E25: 66.3× vs 66.2×
- Причина расхождения: (a) мало данных/эпох, (b) маленький predictor hidden=128
- Ф18 опровергнут, Ф36 подтверждён

---

## НЕРЕШЁННЫЕ ВОПРОСЫ (приоритет)

1. ~~Является ли отбор (feature selection), а не фиксация, истинной причиной преимущества prescribed на Push-T?~~ → Частично отвечено: Push-T 5D prescribed (без отбора) работает (Ф36). Отбор не единственный фактор. Но маятники без отбора не работают — нужно понять различие (П1)
2. Почему Push-T 5D prescribed работает, а маятники prescribed на полном входе — нет? (П1) → E16 перезапускается с нормализацией
3. ~~Почему dim sweep (Ф18) и Tier 3 (Ф36) дают разные результаты для dim=5?~~ → РЕШЕНО (П2): E13 был underpowered. E28 подтвердил: prescribed wins 1–11.
4. ~~Почему даже sinθ (same subspace) ухудшает в 4.8×?~~ → Контекст изменился: prescribed побеждает на всех dim; ухудшение = потеря gap magnitude, не проигрыш
5. ~~Воспроизводятся ли факты Ф17–Ф23 на полных данных?~~ → Ф17 обновлён (E28): prescribed_11 теперь побеждает free_11 (42×) при правильном predictor capacity. Ф21–Ф23 (fragility) — нужен перезапуск с max(128,dim*8) predictor.
6. Как prescribed ведёт себя в средах с dim_state > dim_internal, с нормализацией? → Частично: E16 перезапускается

---

## ПРОТОКОЛ

- Факты: только экспериментальные результаты с параметрами
- Гипотезы: с явным статусом (подтверждена / опровергнута / открыта)
- Подтверждение на одной среде ≠ подтверждение в общем случае — указывать среду
- При противоречии между экспериментами — фиксировать как противоречие (П) с возможными объяснениями
- Обновлять при каждом новом эксперименте
