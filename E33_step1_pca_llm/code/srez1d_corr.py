"""
Slice 1d: formal correlation between within-category cohesion
(Slice 1c) and asymmetric loading frequency (Slice 1b).

Takes two vectors (one point per category) and computes
Spearman and Pearson. Recomputed from source data, not from text
files, to stay independent of output formatting.
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
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


def H(cats, n_cats=8):
    counts = Counter(cats)
    n = sum(counts.values())
    probs = [v/n for v in counts.values()]
    if len(probs) == 1:
        return 0.0
    return -sum(p*np.log(p) for p in probs if p > 0) / np.log(n_cats)


import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))
with open(DATA/'step1_pca'/'step1_pca_results.pkl', 'rb') as f:
    s1 = pickle.load(f)
prompts = s1['prompts']
cats = np.array(s1['categories'])
toks = np.array([last_token(p) for p in prompts])
ohe = OneHotEncoder(sparse_output=False).fit_transform(toks.reshape(-1, 1))

models = ['qwen', 'gemma', 'olmo', 'falcon', 'pythia']
all_cats = sorted(set(cats))

# (1) Cohesion per category, averaged over 5 models
cohesion = {c: [] for c in all_cats}
# (2) Asymmetric loading frequency per category, summed over 5 models
asym_count = Counter()

for model in models:
    with open(DATA/f'{model}_results.pkl', 'rb') as f:
        raw = pickle.load(f)
    L = sorted(raw['activations'].keys())[-1]
    Xp = PCA(n_components=40, svd_solver='full').fit_transform(raw['activations'][L])
    Xr = residualize(Xp, ohe)

    # (1) cohesion
    norms = np.linalg.norm(Xr, axis=1, keepdims=True)
    Xn = Xr / (norms + 1e-9)
    for c in all_cats:
        idx = np.where(cats == c)[0]
        sims = [Xn[i] @ Xn[j]
                for i in idx for j in idx if i < j]
        cohesion[c].append(float(np.mean(sims)))

    # (2) asymmetric appearances
    for pc in range(40):
        proj = Xr[:, pc]
        order = np.argsort(proj)
        ct = [cats[i] for i in order[-5:]]
        cb = [cats[i] for i in order[:5]]
        Hp, Hm = H(ct), H(cb)
        if (Hp < 0.3) != (Hm < 0.3):
            if Hp < 0.3:
                pure_cat = Counter(ct).most_common(1)[0][0]
            else:
                pure_cat = Counter(cb).most_common(1)[0][0]
            asym_count[pure_cat] += 1

# Category vectors in fixed order
cohesion_vec = np.array([np.mean(cohesion[c]) for c in all_cats])
asym_vec = np.array([asym_count.get(c, 0) for c in all_cats])

print("Slice 1d: cohesion vs asymmetric loading frequency")
print()
print(f"{'category':<12} {'cohesion':>10} {'asym count':>11}")
print("-" * 36)
for c, coh, ac in zip(all_cats, cohesion_vec, asym_vec):
    print(f"{c:<12} {coh:>10.3f} {ac:>11d}")
print()

s_rho, s_p = spearmanr(cohesion_vec, asym_vec)
p_r, p_p = pearsonr(cohesion_vec, asym_vec)
print(f"Spearman ρ = {s_rho:.3f}  (p = {s_p:.4f})")
print(f"Pearson  r = {p_r:.3f}  (p = {p_p:.4f})")
print(f"N = {len(all_cats)} categories")
