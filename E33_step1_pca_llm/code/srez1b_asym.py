"""ASYMMETRIC PC details on residuals: which category loads, on which pole."""
import pickle, numpy as np
from pathlib import Path
from collections import Counter
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

def H(cats, n_cats=8):
    counts = Counter(cats)
    n = sum(counts.values())
    probs = [v/n for v in counts.values()]
    if len(probs) == 1: return 0.0
    return -sum(p*np.log(p) for p in probs if p>0) / np.log(n_cats)

import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))
with open(DATA/'step1_pca'/'step1_pca_results.pkl','rb') as f: s1 = pickle.load(f)
prompts = s1['prompts']; cats = np.array(s1['categories'])
toks = np.array([last_token(p) for p in prompts])
ohe = OneHotEncoder(sparse_output=False).fit_transform(toks.reshape(-1,1))

# Which category loads on asymmetric PCs, and how many times
print("="*78)
print("Which categories load asymmetrically on residuals")
print("="*78)

global_counter = Counter()
for model in ['qwen','gemma','olmo','falcon','pythia']:
    with open(DATA/f'{model}_results.pkl','rb') as f: raw = pickle.load(f)
    L = sorted(raw['activations'].keys())[-1]
    Xp = PCA(n_components=40, svd_solver='full').fit_transform(raw['activations'][L])
    Xr = residualize(Xp, ohe)
    
    asyms = []
    for pc in range(40):
        proj = Xr[:, pc]
        order = np.argsort(proj)
        ct = [cats[i] for i in order[-5:]]
        cb = [cats[i] for i in order[:5]]
        Hp, Hm = H(ct), H(cb)
        if (Hp < 0.3) != (Hm < 0.3):  # asymmetry
            # which side is homogeneous
            if Hp < 0.3:
                pure_cat = Counter(ct).most_common(1)[0][0]
                pure_count = Counter(ct).most_common(1)[0][1]
                pure_side = '+'
            else:
                pure_cat = Counter(cb).most_common(1)[0][0]
                pure_count = Counter(cb).most_common(1)[0][1]
                pure_side = '-'
            asyms.append((pc, pure_side, pure_cat, pure_count, Hp, Hm))
    
    print(f"\n{model.upper()}: {len(asyms)} asymmetric PCs")
    for pc, side, cat, cnt, Hp, Hm in asyms:
        print(f"  PC{pc+1:<3} {side}pole = {cnt}/5 {cat:<10} (H+={Hp:.2f} H-={Hm:.2f})")
        global_counter[cat] += 1

print()
print("="*78)
print("Asymmetric loading count per category (across all 5 models)")
print("="*78)
for cat, cnt in global_counter.most_common():
    print(f"  {cat:<12} {cnt} time{'s' if cnt != 1 else ''}")
