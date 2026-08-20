"""
PC poles on last-token residuals. No interpretation.
Per model: top-N PCs by 1D LDA on residuals, and what sits on their poles.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
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
with open(DATA/'step1_pca'/'step1_pca_results.pkl','rb') as f:
    s1 = pickle.load(f)
prompts = s1['prompts']
categories = np.array(s1['categories'])

last_toks = np.array([last_token(p) for p in prompts])
ohe_tok = OneHotEncoder(sparse_output=False).fit_transform(last_toks.reshape(-1,1))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_to_id = {c:i for i,c in enumerate(sorted(set(categories)))}
y_cat = np.array([cat_to_id[c] for c in categories])

# Per model: take final-layer activations (strongest semantics), 
# run PCA, residualize, then inspect top-5 PCs by 1D LDA on residuals
# and for each such PC - what sits on the poles.

OUT_PATH = Path(os.environ.get('YADRO_OUT', '.'))/'PC_polarities_on_residual.txt'
out_lines = []
def w(s=""):
    out_lines.append(s)
    print(s)

for model in ['qwen','gemma','olmo','falcon','pythia']:
    pkl = DATA/f'{model}_results.pkl'
    with open(pkl,'rb') as f:
        raw = pickle.load(f)
    layers = sorted(raw['activations'].keys())
    L_final = layers[-1]
    X_raw = raw['activations'][L_final]
    pca = PCA(n_components=40, svd_solver='full')
    Xp = pca.fit_transform(X_raw)
    evr = pca.explained_variance_ratio_
    Xr = residualize(Xp, ohe_tok)

    # Top-5 PCs by 1D LDA on residuals
    accs = []
    for pc in range(40):
        try:
            a = cross_val_score(LinearDiscriminantAnalysis(),
                                Xr[:, pc:pc+1], y_cat, cv=cv).mean()
        except Exception:
            a = 0.0
        accs.append((pc, a))
    accs.sort(key=lambda x: x[1], reverse=True)
    top5 = accs[:5]

    w("=" * 90)
    w(f"MODEL: {model.upper()} (final layer = {L_final})")
    w("Top-5 PCs by 1D LDA on residuals after last-token residualization")
    w("=" * 90)
    w()
    for rank, (pc, a) in enumerate(top5):
        proj = Xr[:, pc]
        order = np.argsort(proj)
        top_idx = order[-5:][::-1]
        bot_idx = order[:5]
        w(f"━━━ PC{pc+1} (rank {rank+1}, 1D LDA={a:.3f}, orig evr={evr[pc]*100:.2f}%) ━━━")
        w()
        w("  + pole:")
        for i in top_idx:
            w(f"    [{categories[i]:>9}] ({proj[i]:+.3f})  {prompts[i]}")
        w("  − pole:")
        for i in bot_idx:
            w(f"    [{categories[i]:>9}] ({proj[i]:+.3f})  {prompts[i]}")
        # Category summary of poles
        top_cats = [categories[i] for i in top_idx]
        bot_cats = [categories[i] for i in bot_idx]
        w(f"  + categories: {', '.join(top_cats)}")
        w(f"  − categories: {', '.join(bot_cats)}")
        w()
    w()

OUT_PATH.write_text("\n".join(out_lines))
print(f"\nSaved: {OUT_PATH}")
