"""
Slice 1: quantitative criterion for pole pairing.

For each PC of each model:
- Take the 5 nearest prompts from the + and - poles
- Compute the entropy of the category distribution on each pole
- Low entropy on both sides = paired duality
- Low on one side = asymmetric loading  
- High on both = random

Computed on raw final-layer PCA and on residuals after last-token
residualization. The shift is compared.

Shannon entropy normalized by log(8) = log(N_categories).
0 = single category (5/5), 1 = uniform over 5 distinct ones.
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def last_token(prompt):
    p = prompt.rstrip('.,;:!?()[]{}"\' ')
    words = p.split()
    return words[-1].lower() if words else ""

def residualize(X, controls):
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        reg = LinearRegression().fit(controls, X[:, j])
        out[:, j] = X[:, j] - reg.predict(controls)
    return out

def pole_entropy(cats_at_pole, n_categories=8):
    """Normalized Shannon entropy on a pole."""
    counts = Counter(cats_at_pole)
    n = sum(counts.values())
    probs = [c/n for c in counts.values()]
    if len(probs) == 1:
        return 0.0
    H = -sum(p * np.log(p) for p in probs if p > 0)
    H_max = np.log(n_categories)
    return H / H_max

def classify_pole_pair(H_plus, H_minus, low_threshold=0.3):
    """
    Pairing classification.
    low_threshold = 0.3 corresponds to a 4/5 pattern (one category dominates)
    """
    if H_plus < low_threshold and H_minus < low_threshold:
        return "PAIRED"  # both sides homogeneous
    elif H_plus < low_threshold or H_minus < low_threshold:
        return "ASYMMETRIC"  # one side homogeneous
    else:
        return "RANDOM"  # both mixed

# Threshold calibration: 4/5 = entropy log(2 distinct outcomes weighted 4 and 1)
# H = -(4/5 * log(4/5) + 1/5 * log(1/5)) / log(8) ≈ 0.241
# 5/5 = 0
# 3/5+2/5 (two categories, 3:2 dominance) = -(3/5*log(3/5)+2/5*log(2/5))/log(8) ~ 0.323
# 5 distinct = log(5)/log(8) ~ 0.774
# Threshold 0.3: 4/5 ✓, 3/5+2/5 ✗ - 3/5+2/5 is arguably not yet "homogeneous"
# Using 0.3; verified by hand in the output below

import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))
with open(DATA/'step1_pca'/'step1_pca_results.pkl','rb') as f:
    s1 = pickle.load(f)
prompts = s1['prompts']
categories = np.array(s1['categories'])
last_toks = np.array([last_token(p) for p in prompts])
ohe_tok = OneHotEncoder(sparse_output=False).fit_transform(last_toks.reshape(-1,1))
n_cats = len(set(categories))

print(f"Prompts: {len(prompts)}, categories: {n_cats}")
print(f"Entropy threshold calibration (normalized by log({n_cats})):")
test_distrs = [
    [1,1,1,1,1],  # 5/5
    [1,1,1,1,2],  # 4/5
    [1,1,1,2,2],  # 3/5+2/5
    [1,1,2,2,3],  # 2+2+1
    [1,2,3,4,5],  # 5 distinct
]
for d in test_distrs:
    H = pole_entropy(d, n_cats)
    desc = '+'.join(str(c) for c in Counter(d).values())
    print(f"  distribution {desc}: H_norm = {H:.3f}")
print()

results = {}
for model in ['qwen','gemma','olmo','falcon','pythia']:
    pkl = DATA/f'{model}_results.pkl'
    with open(pkl,'rb') as f:
        raw = pickle.load(f)
    layers = sorted(raw['activations'].keys())
    L_final = layers[-1]
    X_raw = raw['activations'][L_final]
    pca = PCA(n_components=40, svd_solver='full')
    Xp = pca.fit_transform(X_raw)
    Xr = residualize(Xp, ohe_tok)

    raw_classes = []
    res_classes = []
    raw_details = []
    res_details = []
    
    for pc in range(40):
        for X_use, store_class, store_det in [(Xp, raw_classes, raw_details), 
                                              (Xr, res_classes, res_details)]:
            proj = X_use[:, pc]
            order = np.argsort(proj)
            top = order[-5:]; bot = order[:5]
            cats_top = [categories[i] for i in top]
            cats_bot = [categories[i] for i in bot]
            H_plus = pole_entropy(cats_top, n_cats)
            H_minus = pole_entropy(cats_bot, n_cats)
            cls = classify_pole_pair(H_plus, H_minus)
            store_class.append(cls)
            store_det.append((pc, H_plus, H_minus, cls, cats_top, cats_bot))

    results[model] = {
        'raw': raw_classes, 'res': res_classes,
        'raw_details': raw_details, 'res_details': res_details,
    }

# Summary: PC count per type
print("="*78)
print("PC count per type out of 40 (entropy threshold 0.3)")
print("="*78)
print(f"\n{'model':<8} | {'PCA raw':<28} | {'PCA on residuals':<28}")
print(f"{'':8} | {'PAIRED':>8} {'ASYM':>6} {'RAND':>6} {'sum':>5} | {'PAIRED':>8} {'ASYM':>6} {'RAND':>6} {'sum':>5}")
print("-"*72)
for m in ['qwen','gemma','olmo','falcon','pythia']:
    rc = Counter(results[m]['raw'])
    rsc = Counter(results[m]['res'])
    print(f"{m:<8} | "
          f"{rc.get('PAIRED',0):>8} {rc.get('ASYMMETRIC',0):>6} {rc.get('RANDOM',0):>6} "
          f"{sum(rc.values()):>5} | "
          f"{rsc.get('PAIRED',0):>8} {rsc.get('ASYMMETRIC',0):>6} {rsc.get('RANDOM',0):>6} "
          f"{sum(rsc.values()):>5}")

# Top-10 PAIRED PCs on residuals (by sum (1-H+) + (1-H-))
print()
print("="*78)
print("Top paired PCs on residuals (most symmetrically homogeneous)")
print("="*78)
for m in ['qwen','gemma','olmo','falcon','pythia']:
    paired = [d for d in results[m]['res_details'] if d[3] == 'PAIRED']
    # sort: lower total entropy = "cleaner" pairing
    paired.sort(key=lambda d: d[1]+d[2])
    print(f"\n{m.upper()}: {len(paired)} paired PCs out of 40 on residuals")
    for d in paired[:5]:
        pc, Hp, Hm, cls, ct, cb = d
        ct_summary = '+'.join(f'{n}{c[:3]}' for c,n in Counter(ct).most_common())
        cb_summary = '+'.join(f'{n}{c[:3]}' for c,n in Counter(cb).most_common())
        print(f"  PC{pc+1:<3} H+={Hp:.2f} H-={Hm:.2f}  +[{ct_summary}] vs -[{cb_summary}]")

# Shift from raw to residuals
print()
print("="*78)
print("Distribution change after residualization")
print("="*78)
for m in ['qwen','gemma','olmo','falcon','pythia']:
    rc = Counter(results[m]['raw'])
    rsc = Counter(results[m]['res'])
    d_paired = rsc.get('PAIRED',0) - rc.get('PAIRED',0)
    d_asym = rsc.get('ASYMMETRIC',0) - rc.get('ASYMMETRIC',0)
    d_rand = rsc.get('RANDOM',0) - rc.get('RANDOM',0)
    print(f"  {m:<8} ΔPAIRED={d_paired:+d}  ΔASYM={d_asym:+d}  ΔRAND={d_rand:+d}")

# Save
import json
out = {}
for m, r in results.items():
    out[m] = {
        'raw_summary': dict(Counter(r['raw'])),
        'res_summary': dict(Counter(r['res'])),
    }
print()
print(f"\nRaw statistics: {json.dumps(out, indent=2)}")
