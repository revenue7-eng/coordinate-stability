"""
Slice 1c: within-category cosine cohesion on final-layer residuals.

This computation underpins the Slice 1b result: why code and emotional
(rather than narrative or ethical) load as asymmetric PC poles.

Metric: mean pairwise cosine similarity between prompts of the same category
after last-token residualization. Averaged over 5 models.
Low value = category is diffuse in the model latent space.
High value = category is compact.

Cohesion vs asymmetric loading count (Slice 1b), exact SVD, 2026-08-21:
  emotional  0.128 (most cohesive)  -> 7
  code       0.086                  -> 8
  spatial    0.086                  -> 1
  abstract   0.083                  -> 4
  factual    0.015                  -> 3
  logical    0.002                  -> 2
  narrative -0.000                  -> 0
  ethical   -0.018 (most diffuse)   -> 0

Note: code and spatial differ by 1.3e-4, below the between-model SEM (~1.2e-2).
The rank order of these two is not resolved by the data. See Slice 1d.
"""
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def last_token(prompt: str) -> str:
    p = prompt.rstrip('.,;:!?()[]{}"\' ')
    words = p.split()
    return words[-1].lower() if words else ""


def residualize(X: np.ndarray, controls: np.ndarray) -> np.ndarray:
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        reg = LinearRegression().fit(controls, X[:, j])
        out[:, j] = X[:, j] - reg.predict(controls)
    return out


import os
DATA = Path(os.environ.get('YADRO_DATA', '.'))

with open(DATA / 'step1_pca' / 'step1_pca_results.pkl', 'rb') as f:
    s1 = pickle.load(f)

prompts = s1['prompts']
categories = np.array(s1['categories'])

last_toks = np.array([last_token(p) for p in prompts])
ohe_tok = OneHotEncoder(sparse_output=False).fit_transform(
    last_toks.reshape(-1, 1))

models = ['qwen', 'gemma', 'olmo', 'falcon', 'pythia']

print("Within-category cosine cohesion")
print("on final layer after last-token residualization")
print(f"Averaged over {len(models)} models ({', '.join(models)})")
print()

results = defaultdict(list)
for model in models:
    pkl_path = DATA / f'{model}_results.pkl'
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    L_final = sorted(raw['activations'].keys())[-1]
    X = raw['activations'][L_final]
    Xp = PCA(n_components=40, svd_solver='full').fit_transform(X)
    Xr = residualize(Xp, ohe_tok)

    # Normalize for cosine
    norms = np.linalg.norm(Xr, axis=1, keepdims=True)
    Xn = Xr / (norms + 1e-9)

    for c in sorted(set(categories)):
        idx = np.where(categories == c)[0]
        sims = []
        for i in idx:
            for j in idx:
                if i < j:
                    sims.append(Xn[i] @ Xn[j])
        results[c].append(float(np.mean(sims)))

# Sort by decreasing mean cohesion
ordered = sorted(set(categories), key=lambda c: -np.mean(results[c]))

print(f"{'category':<12} {'avg cos sim':>12} {'std':>8} {'per model':>40}")
print("-" * 76)
for c in ordered:
    vals = results[c]
    per_model = ' '.join(f"{v:+.3f}" for v in vals)
    print(f"{c:<12} {np.mean(vals):>12.3f} {np.std(vals):>8.3f}   {per_model}")
