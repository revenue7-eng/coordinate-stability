#!/usr/bin/env python3
"""Variance decomposition for E41 and the encoder-level correlations.

Two-way crossed random effects, encoder x head, one observation per cell.
Components from expected mean squares; negative estimates are truncated to zero.
Encoder levels are cell means over the five heads, i.e. the head noise averaged
out, and are matched against R2_readout / eff_rank from E40 (enc seeds 1..8 are
bit-identical to E40 inits 1..8, verified by the acceptance cell).
"""
import json, itertools, math
import numpy as np

G = json.load(open("/mnt/d/coordinate-stability/E41_variance_decomp/results/grid.json"))
S = json.load(open("/mnt/d/coordinate-stability/E40_init_sweep/results/sweep.json"))
assert G["acceptance"]["bit_exact"], "acceptance cell is not bit-exact"
ES, HS = G["enc_seeds"], G["head_seeds"]
assert len(G["cells"]) == len(ES)*len(HS), f"incomplete grid: {len(G['cells'])}"

def mat(metric, log=False):
    X = np.array([[G["cells"][f"e{e}_h{h}"][metric] for h in HS] for e in ES])
    return np.log(X) if log else X

def decomp(X):
    a, b = X.shape; gm = X.mean()
    MSa = b*((X.mean(1)-gm)**2).sum()/(a-1)
    MSb = a*((X.mean(0)-gm)**2).sum()/(b-1)
    R = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm
    MSr = (R**2).sum()/((a-1)*(b-1))
    va, vb, vr = max((MSa-MSr)/b, 0), max((MSb-MSr)/a, 0), MSr
    t = va+vb+vr
    return va/t, vb/t, vr/t, MSa/MSr, MSb/MSr

print(f"{'metric':>12} {'enc':>6} {'head':>6} {'resid':>6} {'F_enc':>9} {'F_head':>7}")
for m in ("best_vp", "final_vp", "mean_last3"):
    for log in (False, True):
        e, h, r, fa, fb = decomp(mat(m, log))
        print(f"{m+(' log' if log else ''):>12} {e:>6.3f} {h:>6.3f} {r:>6.3f} {fa:>9.1f} {fb:>7.2f}")

X = mat("best_vp")
lev = X.mean(1)
print("\nencoder levels (mean over heads):")
print(f"  {'enc':>4} {'level':>10} {'within':>8} {'R2_readout':>11} {'eff_rank':>9}")
r2 = np.array([S["inits"][str(e)]["r2_readout"] for e in ES])
er = np.array([S["inits"][str(e)]["eff_rank"] for e in ES])
for i, e in enumerate(ES):
    print(f"  {e:>4} {lev[i]:>10.6f} {X[i].max()/X[i].min():>7.2f}x "
          f"{r2[i]:>11.4f} {er[i]:>9.4f}")
print(f"  spread of levels: {lev.max()/lev.min():.2f}x   "
      f"(E40 raw spread over inits 1..8: "
      f"{max(S['inits'][str(e)]['best_vp'] for e in ES)/min(S['inits'][str(e)]['best_vp'] for e in ES):.2f}x)")

def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a-a.mean())*(b-b.mean())).sum() /
                 math.sqrt(((a-a.mean())**2).sum()*((b-b.mean())**2).sum()))
def rk(v):
    o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(1, len(v)+1); return r
def perm_p(a, b):
    obs = abs(pear(rk(a), rk(b))); n = len(a); c = t = 0
    for p in itertools.permutations(range(n)):
        t += 1; c += abs(pear(rk(np.asarray(a)[list(p)]), rk(b))) >= obs - 1e-12
    return c/t

print("\nencoder level vs candidate predictors (n=8, head noise averaged out):")
for nm, v in (("R2_readout", r2), ("eff_rank", er)):
    print(f"  {nm:>11}: pearson {pear(lev, v):+.4f}  spearman {pear(rk(lev), rk(v)):+.4f} "
          f" exact p {perm_p(lev, v):.4f}")
print(f"  {'R2 vs rank':>11}: pearson {pear(r2, er):+.4f}")
e40 = np.array([S["inits"][str(e)]["best_vp"] for e in ES])
print(f"\nE40 best_vp vs E41 encoder level: pearson {pear(e40, lev):+.4f} "
      f"spearman {pear(rk(e40), rk(lev)):+.4f}")
