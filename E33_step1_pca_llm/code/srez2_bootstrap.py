"""
Slice 2: bootstrap stability of nodes.

Idea: if a category forms a "node" on a PC pole because it is structurally
cohesive in the residual stream, the node should survive prompt resampling.

Metric: for each model and each loading category, how often
that category appears as a pole of some PC across bootstrap samples.

Protocol:
- 30 bootstrap runs
- Each: 70% of prompts sampled at random
- PCA + last-token residualization (on that subsample)
- Among the top-15 PCs, look for asymmetric nodes (one side >=4/5 single category)
- Count how many runs each category appears in at least once
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def last_token(p):
    p = p.rstrip('.,;:!?()[]{}"\' ')
    w = p.split()
    return w[-1].lower() if w else ""

def residualize(X, c):
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        r = LinearRegression().fit(c, X[:, j])
        out[:, j] = X[:, j] - r.predict(c)
    return out

import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))
with open(DATA/'step1_pca'/'step1_pca_results.pkl','rb') as f:
    s1 = pickle.load(f)
prompts_full = np.array(s1['prompts'])
cats_full = np.array(s1['categories'])
N_full = len(prompts_full)

rng = np.random.RandomState(42)
N_BOOT = 20
SUBSAMPLE_FRAC = 0.7
N_SUB = int(N_full * SUBSAMPLE_FRAC)
N_PC_CHECK = 15  # inspect the top-15 PCs after PCA in each bootstrap
N_PCA_COMP = 25  # 25 instead of 40 for speed

# Which categories appear as poles, and how often
def find_asymmetric_categories(Xr, cats_local, top_k=15, threshold_count=4):
    """Returns categories appearing as a PC pole, with the side noted."""
    found = []
    for pc in range(min(top_k, Xr.shape[1])):
        proj = Xr[:, pc]
        order = np.argsort(proj)
        ct = [cats_local[i] for i in order[-5:]]
        cb = [cats_local[i] for i in order[:5]]
        for cats_at_pole, side in [(ct, '+'), (cb, '-')]:
            counter = Counter(cats_at_pole)
            top_cat, top_count = counter.most_common(1)[0]
            if top_count >= threshold_count:
                # Check the other side is NOT the same category
                # (to exclude paired PCs)
                other_cats = cb if side == '+' else ct
                other_counter = Counter(other_cats)
                other_top_cat, other_top_count = other_counter.most_common(1)[0]
                if other_top_count >= threshold_count and other_top_cat == top_cat:
                    continue  # both sides identical - anomalous, not counted
                found.append((pc, side, top_cat, top_count))
    return found

print("Slice 2: bootstrap stability of nodes")
print(f"  N_full={N_full}, N_subsample={N_SUB}, N_bootstrap={N_BOOT}")
print(f"  node threshold: >=4/5 single category on a pole of one of the top-{N_PC_CHECK} PCs")
print()

# For each model:
# (1) baseline on the full sample - which categories were nodes
# (2) bootstrap - appearance counter per category
all_results = {}

for model in ['qwen','gemma','olmo','falcon','pythia']:
    pkl = DATA/f'{model}_results.pkl'
    with open(pkl,'rb') as f:
        raw = pickle.load(f)
    L = sorted(raw['activations'].keys())[-1]
    X_full = raw['activations'][L]

    # Baseline: on the full sample
    toks_full = np.array([last_token(p) for p in prompts_full])
    ohe_full = OneHotEncoder(sparse_output=False).fit_transform(toks_full.reshape(-1,1))
    Xp_full = PCA(n_components=N_PCA_COMP, svd_solver='full').fit_transform(X_full)
    Xr_full = residualize(Xp_full, ohe_full)
    baseline_nodes = find_asymmetric_categories(Xr_full, cats_full, N_PC_CHECK)
    baseline_cats = Counter([c for _,_,c,_ in baseline_nodes])

    # Bootstrap
    boot_cats = Counter()
    boot_n_nodes = []
    for b in range(N_BOOT):
        idx = rng.choice(N_full, size=N_SUB, replace=False)
        X_sub = X_full[idx]
        cats_sub = cats_full[idx]
        prompts_sub = prompts_full[idx]
        toks_sub = np.array([last_token(p) for p in prompts_sub])
        # A subsample may lack some last_token present in the full set - that is fine
        ohe_sub = OneHotEncoder(sparse_output=False).fit_transform(toks_sub.reshape(-1,1))
        # PCA needs >= n_components samples; N_SUB=56 >= 40
        n_comp = min(N_PCA_COMP, N_SUB - 1)
        Xp_sub = PCA(n_components=n_comp, svd_solver='full').fit_transform(X_sub)
        Xr_sub = residualize(Xp_sub, ohe_sub)
        nodes_b = find_asymmetric_categories(Xr_sub, cats_sub, N_PC_CHECK)
        # Count unique categories appearing in this bootstrap run
        unique_cats_in_boot = set(c for _,_,c,_ in nodes_b)
        for c in unique_cats_in_boot:
            boot_cats[c] += 1
        boot_n_nodes.append(len(nodes_b))

    all_results[model] = {
        'baseline_cats': baseline_cats,
        'boot_cats': boot_cats,
        'boot_avg_n_nodes': np.mean(boot_n_nodes),
        'boot_std_n_nodes': np.std(boot_n_nodes),
    }

print("="*82)
print(f"{'model':<8} {'category':<11} {'baseline':<10} {'boot %':<10} {'stable?':<12}")
print("="*82)
all_cats = ['code','emotional','abstract','factual','spatial','logical','narrative','ethical']
for model in ['qwen','gemma','olmo','falcon','pythia']:
    r = all_results[model]
    print(f"{model:<8}                                                              "
          f"(avg nodes per bootstrap: {r['boot_avg_n_nodes']:.1f}±{r['boot_std_n_nodes']:.1f})")
    for c in all_cats:
        bl = r['baseline_cats'].get(c, 0)
        bp = r['boot_cats'].get(c, 0)
        pct = 100 * bp / N_BOOT
        if bl > 0 and pct >= 70:
            verdict = "✓ stable"
        elif bl > 0 and pct >= 30:
            verdict = "~ partial"
        elif bl > 0 and pct < 30:
            verdict = "✗ unstable"
        elif bl == 0 and pct >= 30:
            verdict = "appears (absent in baseline)"
        else:
            verdict = "—"
        if bl > 0 or pct >= 30:
            print(f"  {'':<6} {c:<11} {bl:<10} {pct:<10.0f} {verdict:<12}")
print()

# Stability summary
print("="*82)
print("Summary: categories that stably form nodes (>=70% of bootstraps)")
print("="*82)
robust = defaultdict(list)
for model in ['qwen','gemma','olmo','falcon','pythia']:
    r = all_results[model]
    for c in all_cats:
        bp = r['boot_cats'].get(c, 0)
        pct = 100 * bp / N_BOOT
        if pct >= 70:
            robust[c].append(model)
print()
for c in all_cats:
    if c in robust:
        print(f"  {c:<12} stable in: {', '.join(robust[c])} ({len(robust[c])}/5 models)")
    else:
        print(f"  {c:<12} not stable in any model")
