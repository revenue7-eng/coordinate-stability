#!/usr/bin/env python3
"""E38 window-sensitivity check.

Same linear-vs-step comparison as E32's analyze_shape.py, but evaluated over
several windows instead of the single hardcoded 0.25-0.60 band. Purpose: test
whether the STEP verdict on full-budget data is a property of the curve or a
property of the legacy window (which now straddles the rise and the plateau).

Nothing here changes E32's analyzer; that stays the reference for comparability.
"""
import json, glob, os, numpy as np

OUT = '/mnt/d/coordinate-stability/E38_subepoch_freeze_full/results'
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0]

WINDOWS = {
    'legacy 0.25-0.60 (E31/E32 band)': [f for f in GRID if 0.25 <= f <= 0.60],
    'rise 0.00-0.40':                  [f for f in GRID if 0.00 <= f <= 0.40],
    'rise 0.00-0.45':                  [f for f in GRID if 0.00 <= f <= 0.45],
    'full 0.00-1.00':                  list(GRID),
    'plateau 0.45-1.00':               [f for f in GRID if 0.45 <= f <= 1.00],
}

seeds = {}
for p in sorted(glob.glob(OUT + '/seed_*.json')):
    d = json.load(open(p))
    seeds[d['seed']] = d
print('seeds loaded:', sorted(seeds))
print('points per seed:', len(seeds[list(seeds)[0]]['sweep']))


def analyze(band):
    bk = ['%.2f' % f for f in band]
    mono_full = 0
    dips_total = 0
    rows = []
    for s in sorted(seeds):
        y = [seeds[s]['sweep'][k] for k in bk]
        dips = sum(1 for i in range(1, len(y)) if y[i] < y[i - 1])
        rows.append((s, dips))
        if dips == 0:
            mono_full += 1
        dips_total += dips

    X, Y = [], []
    for s in sorted(seeds):
        y = np.array([seeds[s]['sweep'][k] for k in bk], float)
        yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
        X += list(band)
        Y += list(yn)
    X = np.array(X); Y = np.array(Y)

    A = np.vstack([X, np.ones_like(X)]).T
    coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    ss_lin = float(((Y - A @ coef) ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2_lin = 1 - ss_lin / ss_tot

    ss_step = np.inf; best_bp = None
    for bp in range(1, len(band)):
        thr = band[bp]
        left = Y[X < thr]; right = Y[X >= thr]
        if len(left) == 0 or len(right) == 0:
            continue
        ss = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if ss < ss_step:
            ss_step = float(ss); best_bp = thr

    ratio = ss_step / ss_lin
    if ratio >= 1.3 and r2_lin >= 0.75 and mono_full >= 3:
        verdict = 'SLOPE'
    elif ratio <= 0.7:
        verdict = 'STEP'
    else:
        verdict = 'INCONCLUSIVE'
    return dict(n=len(band), mono=mono_full, dips=dips_total, r2=r2_lin,
                ss_lin=ss_lin, ss_step=ss_step, ratio=ratio, bp=best_bp,
                verdict=verdict, rows=rows)


print()
print('=' * 76)
print('WINDOW SENSITIVITY OF THE LINEAR-VS-STEP VERDICT')
print('=' * 76)
print(f"{'window':<34}{'pts':>4}{'mono':>6}{'R2_lin':>9}{'ratio':>8}{'bp':>6}  verdict")
print('-' * 76)
for name, band in WINDOWS.items():
    r = analyze(band)
    bp = '' if r['bp'] is None else f"{r['bp']:.2f}"
    print(f"{name:<34}{r['n']:>4}{r['mono']:>4}/5{r['r2']:>9.3f}"
          f"{r['ratio']:>8.2f}{bp:>6}  {r['verdict']}")

print()
print('Per-seed curves (full grid):')
for s in sorted(seeds):
    y = [seeds[s]['sweep']['%.2f' % f] for f in GRID]
    print(f'  seed {s:>4}: ' + ' '.join(f'{v:.4f}' for v in y))

print()
print('Onset damage (f=0.00 -> f=0.25) and anchor (f=1.00 / f=0.00):')
for s in sorted(seeds):
    sw = seeds[s]['sweep']
    f0, f25, f1 = sw['0.00'], sw['0.25'], sw['1.00']
    print(f'  seed {s:>4}: onset x{f25 / f0:6.1f}   anchor x{f1 / f0:6.1f}   '
          f'prescribed={seeds[s].get("prescribed", float("nan")):.5f}')

print()
print('Plateau onset per seed (first f where the curve stays within 5% of its max):')
for s in sorted(seeds):
    y = np.array([seeds[s]['sweep']['%.2f' % f] for f in GRID])
    ymax = y.max()
    idx = next((i for i in range(len(GRID)) if (y[i:] >= 0.95 * ymax).all()), None)
    print(f'  seed {s:>4}: f={GRID[idx]:.2f}' if idx is not None else f'  seed {s:>4}: none')
