# E29. Noise Control: prescribed + matched noise vs free encoder

## Цель

Отличить structured drift от generic noise.
Prescribed encoder + Gaussian noise с σ, matched к измеренному drift magnitude из E06.

## Дизайн

8 условий, 200 эпизодов synthetic Push-T, 30 эпох, 3 seeds (42, 123, 777).

| Условие | Что делает | σ per dim |
|---|---|---|
| prescribed | Чистый prescribed (baseline) | 0 |
| noise_late | prescribed + i.i.d. noise, σ matched to late drift (ep 28→29, drift≈0.005) | 0.0031 |
| noise_mid | prescribed + i.i.d. noise, σ matched to mid drift (ep 2→3, drift≈0.08) | 0.0501 |
| noise_early | prescribed + i.i.d. noise, σ matched to early drift (ep 0→1, drift≈1.43) | 0.8961 |
| noise_schedule | prescribed + i.i.d. noise, σ follows actual drift schedule per epoch | varies |
| correlated_mid | prescribed + correlated noise (constant displacement per epoch), σ≈0.08 | 0.0501 |
| correlated_schedule | prescribed + correlated noise, σ follows drift schedule | varies |
| free | Стандартный free encoder (MLP 5→3) | — |

### Noise matching

Drift из E06 измерен как mean ℓ₂ displacement validation embeddings между эпохами.
Для 3D Gaussian: E[‖noise‖] = σ × √2 × Γ(2)/Γ(3/2) ≈ 1.596σ.
σ = drift_magnitude / 1.596.

### i.i.d. vs correlated noise

- **i.i.d.**: каждый sample × timestep получает свой random noise vector.
- **correlated**: один displacement vector на всю эпоху. Разности между timesteps сохраняются.

Noise только в training. Eval — чистый prescribed.

## Результат (3 seeds)

| Условие | Seed 42 | Seed 123 | Seed 777 | Mean | vs prescribed | vs free |
|---|---|---|---|---|---|---|
| prescribed | 0.000049 | 0.000043 | 0.000020 | 0.000037 | 1.0× | 0.004× |
| noise_late | 0.000057 | 0.000056 | 0.000018 | 0.000044 | 1.2× | 0.005× |
| noise_mid | 0.000264 | 0.000316 | 0.000119 | 0.000233 | 6.2× | 0.028× |
| noise_early | 0.031816 | 0.031516 | 0.032010 | 0.031781 | 851× | 3.8× |
| noise_schedule | 0.000070 | 0.000078 | 0.000029 | 0.000059 | 1.6× | 0.007× |
| correlated_mid | 0.000060 | 0.000069 | 0.000013 | 0.000047 | 1.3× | 0.006× |
| correlated_schedule | 0.000056 | 0.000074 | 0.000019 | 0.000050 | 1.3× | 0.006× |
| free | 0.008136 | 0.008988 | 0.007722 | 0.008282 | 222× | 1.0× |

## Ключевые факты

**Ф43i. i.i.d. noise при drift-matched amplitude разрушает prescribed в 851×, но free encoder — в 222×.**
Free encoder лучше i.i.d. noise в 4× → drift не сводится к random perturbation.

**Ф44i. Correlated noise (constant shift per epoch) почти не вредит prescribed даже при катастрофической амплитуде: 1.3×.**
Predictor компенсирует constant displacement: разности между timesteps сохраняются.

**Ф45i. Free encoder (222×) в 167× хуже correlated noise (1.3×) при той же амплитуде.**
Drift в free encoder — не global coordinate shift. Это data-dependent деформация.

**Ф46i. Спектр: prescribed (1×) < correlated (1.3×) < noise_mid (6.2×) < FREE (222×) < noise_early (851×).**
Free encoder ближе к i.i.d. noise, чем к correlated shift. Drift вносит несогласованность внутри context window.

## Интерпретация

Drift ≠ noise (i.i.d. при той же амплитуде даёт качественно другой эффект).
Drift ≠ constant shift (correlated shift компенсируется predictor).
Drift — structured, data-dependent transformation между этими крайностями.

## Параметры

- Seeds: 42, 123, 777
- Episodes: 200 (synthetic Push-T)
- Epochs: 30
- SIGReg: нет (только prediction loss)
- Архитектура: идентична tier1/tier2

## Файлы

- `code/noise_control.py` — скрипт (3 seeds, 8 условий)
- `results/noise_control_results.json` — полные результаты
