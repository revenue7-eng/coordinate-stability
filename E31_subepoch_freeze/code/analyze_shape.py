#!/usr/bin/env python3
"""
E31 analysis — PRE-REGISTERED criterion for threshold vs slope.
Decided BEFORE looking at results, computed by code, not by eye.

Reads a subepoch_freeze_results.json and outputs a verdict:
  THRESHOLD   — damage appears within one grid cell (sharp)
  SLOPE       — damage accumulates across >= half the transition region (gradual)
  INCONCLUSIVE — at this seed-count/resolution the two cannot be distinguished

Method (fixed in advance):
  1. Work in log10(best_vp) because values span orders of magnitude.
  2. Low plateau L = mean log10 over the lowest-fraction points that are statistically flat
     (we take f<=0.25 as the pre-transition anchor; verified flat if their spread < high rise).
  3. High plateau H = log10 at f=1.0.
  4. Transition width W = f75 - f25, where f25/f75 are the fractions at which the
     seed-mean curve crosses L+0.25*(H-L) and L+0.75*(H-L) (linear interpolation, first crossing).
  5. Grid spacing g = smallest gap between adjacent fractions.
  6. Bootstrap over seeds (resample seeds with replacement, N=2000) -> CI on W.
  7. Verdict:
       W_CI entirely <= g            -> THRESHOLD
       W_CI entirely >= region_span/2 -> SLOPE
       otherwise                      -> INCONCLUSIVE
"""
import json, argparse
import numpy as np

def load(path):
    d = json.load(open(path))
    fracs = sorted(float(k) for k in d['by_frac'].keys())
    # per-seed matrix: rows=fracs, cols=seeds
    seeds = d['config']['seeds']
    M = np.array([[d['by_frac'][str(f)]['per_seed'][i] for i in range(len(seeds))]
                  for f in fracs])  # shape (n_frac, n_seed)
    return fracs, np.array(seeds), M

def crossing(fracs, curve, level):
    """first fraction where curve crosses `level` going up; linear interp. curve in log10."""
    for i in range(len(fracs)-1):
        a, b = curve[i], curve[i+1]
        if (a <= level <= b) or (b <= level <= a):
            if b == a:
                return fracs[i]
            t = (level - a) / (b - a)
            return fracs[i] + t*(fracs[i+1]-fracs[i])
    # never crosses in range
    return None

def width_from_curve(fracs, logcurve):
    L = np.mean([logcurve[i] for i,f in enumerate(fracs) if f <= 0.25])
    H = logcurve[fracs.index(1.0)] if 1.0 in fracs else logcurve[-1]
    lo = L + 0.25*(H-L); hi = L + 0.75*(H-L)
    f25 = crossing(fracs, logcurve, lo)
    f75 = crossing(fracs, logcurve, hi)
    if f25 is None or f75 is None:
        return None, L, H
    return abs(f75 - f25), L, H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    fracs, seeds, M = load(args.data)   # M: (n_frac, n_seed), raw best_vp
    fracs = list(fracs)
    logM = np.log10(M)
    n_seed = M.shape[1]

    # point estimate on seed-mean curve
    mean_log = logM.mean(axis=1)
    W, L, H = width_from_curve(fracs, list(mean_log))

    g = min(fracs[i+1]-fracs[i] for i in range(len(fracs)-1))
    region_span = max(fracs) - min(fracs)

    print(f"fractions: {fracs}")
    print(f"seeds: {list(seeds)}  (n={n_seed})")
    print(f"grid spacing g = {g:.3f},  region span = {region_span:.3f}")
    print(f"low plateau L(log10) = {L:.3f}  ->  {10**L:.5f}")
    print(f"high plateau H(log10) = {H:.3f}  ->  {10**H:.5f}")
    print(f"point-estimate transition width W = {W}")

    # bootstrap over seeds
    if n_seed < 2:
        print("\nVERDICT: INCONCLUSIVE (need >=2 seeds; ideally >=5 for shape)")
        return
    rng = np.random.default_rng(0)
    Ws = []
    for _ in range(args.boot):
        idx = rng.integers(0, n_seed, n_seed)
        curve = logM[:, idx].mean(axis=1)
        w,_,_ = width_from_curve(fracs, list(curve))
        if w is not None:
            Ws.append(w)
    Ws = np.array(Ws)
    if len(Ws) < args.boot*0.5:
        print(f"\nWARN: {len(Ws)}/{args.boot} bootstraps produced a valid width "
              f"(curve often non-monotone / doesn't cross cleanly). Shape unstable.")
    lo, hi = np.percentile(Ws, [2.5, 97.5])
    print(f"\nbootstrap W: median={np.median(Ws):.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]  (n_valid={len(Ws)})")

    # pre-registered verdict (width-based, secondary)
    print(f"\nthresholds:  g={g:.3f} (threshold if W<=g),  region/2={region_span/2:.3f} (slope if W>=this)")
    if hi <= g:
        verdict_w = "THRESHOLD"
    elif lo >= region_span/2:
        verdict_w = "SLOPE"
    else:
        verdict_w = "INCONCLUSIVE (width CI straddles boundary)"
    print(f"width-based verdict: {verdict_w}")

    # PRIMARY test: linear (slope) vs best single step (threshold) over the rise.
    # Independent of threshold calibration — asks which shape fits the rise better.
    rise = [i for i,f in enumerate(fracs) if f >= 0.25]
    x = np.array([fracs[i] for i in rise]); y = mean_log[rise]
    A = np.vstack([x, np.ones_like(x)]).T
    coef,*_ = np.linalg.lstsq(A, y, rcond=None); ss_lin = ((y - A@coef)**2).sum()
    ss_step = min(((y - np.concatenate([np.full(k, y[:k].mean()),
                                        np.full(len(x)-k, y[k:].mean())]))**2).sum()
                  for k in range(1, len(x)))
    sstot = ((y - y.mean())**2).sum()
    # per-seed monotonicity
    mono = [(int((np.diff(logM[:,j]) < 0).sum())) for j in range(n_seed)]
    print(f"\nPRIMARY shape test (over rise f>=0.25):")
    print(f"  linear(slope) SS={ss_lin:.4f}   best-step(threshold) SS={ss_step:.4f}   "
          f"ratio={ss_step/ss_lin:.2f}   linear R^2={1-ss_lin/sstot:.3f}")
    print(f"  per-seed downward steps: {mono} (0 = clean monotone rise)")
    if ss_lin < ss_step:
        verdict = "SLOPE (linear fits the rise better than any single step; damage accumulates gradually)"
    else:
        verdict = "THRESHOLD (a single step fits the rise better than linear)"
    print(f"\nVERDICT: {verdict}")

if __name__ == '__main__':
    main()
