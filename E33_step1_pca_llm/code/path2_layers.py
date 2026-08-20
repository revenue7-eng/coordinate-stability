"""
Path 2: layer-by-layer analysis on gemma/olmo/falcon/pythia
Repeats srez_by_layer from analiz_kod.py for all 4 models.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))
OUT = Path(os.environ.get('YADRO_OUT', '.'))
with open(DATA/'step1_pca'/'step1_pca_results.pkl','rb') as f:
    s1 = pickle.load(f)
prompts = s1['prompts']
categories = np.array(s1['categories'])

last_toks = np.array([last_token(p) for p in prompts])
ohe_tok = OneHotEncoder(sparse_output=False).fit_transform(last_toks.reshape(-1,1))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_to_id = {c:i for i,c in enumerate(sorted(set(categories)))}
y_cat = np.array([cat_to_id[c] for c in categories])

print(f"Prompts: {len(prompts)}, categories: {len(cat_to_id)}, "
      f"unique last tokens: {len(set(last_toks))}")
print()

def by_layer(model):
    pkl = DATA/f'{model}_results.pkl'
    with open(pkl,'rb') as f:
        raw = pickle.load(f)
    print("="*78)
    print(f"LAYER-BY-LAYER ANALYSIS: {model.upper()}")
    print("="*78)
    layers = sorted(raw['activations'].keys())
    print(f"\n{'layer':<7}{'PC1 evr':<10}{'avg R²(tok)':<14}"
          f"{'orig 7':<9}{'resid 7':<10}{'orig 15':<10}{'resid 15':<11}")
    print("-"*70)
    rows = []
    for L in layers:
        X_raw = raw['activations'][L]
        pca = PCA(n_components=40, svd_solver='full')
        Xp = pca.fit_transform(X_raw)
        evr = pca.explained_variance_ratio_
        r2s = []
        for pc in range(7):
            r2 = LinearRegression().fit(ohe_tok, Xp[:,pc]).score(ohe_tok, Xp[:,pc])
            r2s.append(r2)
        Xr = residualize(Xp, ohe_tok)
        a7  = cross_val_score(LinearDiscriminantAnalysis(), Xp[:,:7],  y_cat, cv=cv).mean()
        a7r = cross_val_score(LinearDiscriminantAnalysis(), Xr[:,:7],  y_cat, cv=cv).mean()
        a15 = cross_val_score(LinearDiscriminantAnalysis(), Xp[:,:15], y_cat, cv=cv).mean()
        a15r= cross_val_score(LinearDiscriminantAnalysis(), Xr[:,:15], y_cat, cv=cv).mean()
        print(f"{L:<7}{evr[0]*100:<9.2f} {np.mean(r2s):<13.3f} "
              f"{a7:<8.3f} {a7r:<9.3f} {a15:<9.3f} {a15r:<10.3f}")
        rows.append({'layer':L,'pc1_evr':evr[0],'avg_r2_tok':float(np.mean(r2s)),
                     'orig7':a7,'resid7':a7r,'orig15':a15,'resid15':a15r})
    print()
    return rows

all_results = {}
for m in ['qwen','gemma','olmo','falcon','pythia']:
    all_results[m] = by_layer(m)

# Summary table: resid_15 by normalized depth (0..1)
print("="*78)
print("SUMMARY: resid 15 PC by normalized depth")
print("="*78)
print(f"\n{'depth':<8}{'qwen':<8}{'gemma':<8}{'olmo':<8}{'falcon':<8}{'pythia':<8}")
# 5 layers per model - indices 0..4
labels = ['0.0 (0)', '0.25', '0.5 (mid)', '0.75', '1.0 (last)']
for i, lab in enumerate(labels):
    row = [lab]
    for m in ['qwen','gemma','olmo','falcon','pythia']:
        row.append(f"{all_results[m][i]['resid15']:.3f}")
    print(f"{row[0]:<8}{row[1]:<8}{row[2]:<8}{row[3]:<8}{row[4]:<8}{row[5]:<8}")

print(f"\n{'depth':<8}{'qwen':<8}{'gemma':<8}{'olmo':<8}{'falcon':<8}{'pythia':<8}  (avg R²(tok) over top-7 PCs)")
for i, lab in enumerate(labels):
    row = [lab]
    for m in ['qwen','gemma','olmo','falcon','pythia']:
        row.append(f"{all_results[m][i]['avg_r2_tok']:.3f}")
    print(f"{row[0]:<8}{row[1]:<8}{row[2]:<8}{row[3]:<8}{row[4]:<8}{row[5]:<8}")

print(f"\n{'depth':<8}{'qwen':<8}{'gemma':<8}{'olmo':<8}{'falcon':<8}{'pythia':<8}  (orig 15 PC)")
for i, lab in enumerate(labels):
    row = [lab]
    for m in ['qwen','gemma','olmo','falcon','pythia']:
        row.append(f"{all_results[m][i]['orig15']:.3f}")
    print(f"{row[0]:<8}{row[1]:<8}{row[2]:<8}{row[3]:<8}{row[4]:<8}{row[5]:<8}")

# Save
import json
with open(OUT/'path2_results.json','w') as f:
    json.dump(all_results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(f"\nSaved: {OUT/'path2_results.json'}")
