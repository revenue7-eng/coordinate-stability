"""
Slice 1e: robustness of the cohesion vs asymmetric-loading correlation (Ф54).

Supplies the three quantities that srez1d_corr.py does not compute:
  1. permutation null   -- is the correlation produced by the category
                           structure, or by the geometry alone?
  2. leave-one-model-out -- how far does rho travel when one of the five
                           models is dropped?
  3. code/spatial gap    -- the margin that decides the rank of two of the
                           eight categories, against between-model SEM.

Data path is identical to srez1d_corr.py: last layer of each raw
{model}_results.pkl, PCA(n_components=40, svd_solver='full'), residualized
against one-hot last-token. Categories and prompts come from
step1_pca/step1_pca_results.pkl.

On the RNG seed. The ten PCA calls in this experiment deliberately carry NO
seed: they use svd_solver='full', an exact decomposition, where a seed could
only freeze one particular draw of an approximation. The seed below is a
different object. The null distribution is a Monte Carlo sample -- 200 draws
from the 8! possible label assignments -- so the seed fixes reproducibility of
the sample, not a numerical error. Do not remove it, and do not add seeds to
the PCA calls by analogy.

Usage (from a results/ directory):
    YADRO_DATA=~/e33work/yadro_phase2 python ../code/srez1e_robustness.py

Prints to stdout; redirect to results/srez1e_output.txt.
"""

import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

MODELS = ['qwen', 'gemma', 'olmo', 'falcon', 'pythia']
N_COMP = 40
N_PERM = 200
PERM_SEED = 42
H_THRESHOLD = 0.3
TOP_K = 5

# Reference value from the committed artifact results/srez1d_output.txt
# (md5 c4fa3939dfe08672670721fa3f3afb70). Used as a method calibration check:
# this script reimplements the pipeline, so it must land on the same number.
REFERENCE_RHO = 0.8264
REFERENCE_TOL = 5e-4


def last_token(prompt):
    prompt = prompt.rstrip('.,;:!?()[]{}"\' ')
    words = prompt.split()
    return words[-1].lower() if words else ""


def residualize(X, controls):
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        out[:, j] = X[:, j] - LinearRegression().fit(controls, X[:, j]).predict(controls)
    return out


def entropy(labels, n_cats=8):
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [v / total for v in counts.values()]
    if len(probs) == 1:
        return 0.0
    return -sum(p * np.log(p) for p in probs if p > 0) / np.log(n_cats)


def cohesion_and_asym(labels, resid, models):
    """Per-category mean cohesion (averaged over models) and asymmetric
    loading counts (summed over models). Mirrors srez1c + srez1b."""
    categories = sorted(set(labels))
    coh = {c: [] for c in categories}
    asym = Counter()
    for m in models:
        Xr = resid[m]
        Xn = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
        for c in categories:
            idx = np.where(labels == c)[0]
            coh[c].append(float(np.mean([Xn[i] @ Xn[j] for i in idx for j in idx if i < j])))
        for pc in range(N_COMP):
            order = np.argsort(Xr[:, pc])
            top = [labels[i] for i in order[-TOP_K:]]
            bottom = [labels[i] for i in order[:TOP_K]]
            h_top, h_bottom = entropy(top), entropy(bottom)
            if (h_top < H_THRESHOLD) != (h_bottom < H_THRESHOLD):
                side = top if h_top < H_THRESHOLD else bottom
                asym[Counter(side).most_common(1)[0][0]] += 1
    coh_mean = np.array([np.mean(coh[c]) for c in categories])
    asym_count = np.array([asym.get(c, 0) for c in categories])
    return categories, coh_mean, asym_count, coh


def main():
    data = Path(os.environ.get('YADRO_DATA', '.'))

    with open(data / 'step1_pca' / 'step1_pca_results.pkl', 'rb') as f:
        s1 = pickle.load(f)
    labels = np.array(s1['categories'])
    tokens = np.array([last_token(p) for p in s1['prompts']])
    ohe = OneHotEncoder(sparse_output=False).fit_transform(tokens.reshape(-1, 1))

    print('=== Slice 1e: robustness of the cohesion / asymmetry correlation ===')
    print(f'prompts             : {len(labels)}')
    print(f'unique last tokens  : {len(set(tokens))}')
    print(f'one-hot controls    : {ohe.shape[1]}')
    print(f'residual DoF        : {len(labels) - ohe.shape[1]}')

    resid = {}
    var_ratio = None
    for m in MODELS:
        with open(data / f'{m}_results.pkl', 'rb') as f:
            raw = pickle.load(f)
        layer = sorted(raw['activations'].keys())[-1]
        Xp = PCA(n_components=N_COMP, svd_solver='full').fit_transform(raw['activations'][layer])
        resid[m] = residualize(Xp, ohe)
        if var_ratio is None:
            var_ratio = float(np.var(resid[m]) / np.var(Xp))
    print(f'residual var / orig : {var_ratio:.4f}   (degeneracy check)')

    categories, coh_mean, asym_count, coh_per_model = cohesion_and_asym(labels, resid, MODELS)
    rho_obs, p_rho = spearmanr(coh_mean, asym_count)
    r_obs, p_r = pearsonr(coh_mean, asym_count)
    print()
    print(f'observed Spearman   : rho = {rho_obs:.4f}  (parametric p = {p_rho:.4f})')
    print(f'observed Pearson    : r   = {r_obs:.4f}  (parametric p = {p_r:.4f})')
    print(f'N (categories)      : {len(categories)}')

    print()
    print('--- CHECK A: permutation null (labels shuffled, geometry fixed) ---')
    rng = np.random.RandomState(PERM_SEED)
    null = []
    for _ in range(N_PERM):
        perm = labels.copy()
        rng.shuffle(perm)
        _, cv, av, _ = cohesion_and_asym(perm, resid, MODELS)
        val = spearmanr(cv, av)[0]
        if not np.isnan(val):
            null.append(float(val))
    null = np.array(null)
    n_ge = int((null >= rho_obs).sum())
    p_perm = (1 + n_ge) / (1 + len(null))
    print(f'draws requested     : {N_PERM}   (seed {PERM_SEED})')
    print(f'draws valid         : {len(null)}   ({N_PERM - len(null)} dropped: NaN, '
          f'a permuted category had constant asym count)')
    print(f'null mean / sd      : {null.mean():.4f} / {null.std():.4f}')
    print(f'null q50/q90/q95    : {np.quantile(null, .5):.4f} / '
          f'{np.quantile(null, .9):.4f} / {np.quantile(null, .95):.4f}')
    print(f'null max            : {null.max():.4f}')
    print(f'null >= observed    : {n_ge} of {len(null)}')
    print(f'p_perm (conservative, (1+k)/(1+n)) : {p_perm:.4f}')

    print()
    print('--- CHECK B: leave-one-model-out ---')
    print(f'{"dropped":<10}{"rho":>9}{"p_rho":>9}{"r":>9}{"p_r":>9}')
    loo_rho, loo_p = [], []
    for drop in MODELS:
        subset = [m for m in MODELS if m != drop]
        _, cv, av, _ = cohesion_and_asym(labels, resid, subset)
        rr, pp = spearmanr(cv, av)
        r2, p2 = pearsonr(cv, av)
        loo_rho.append(rr)
        loo_p.append(pp)
        print(f'{drop:<10}{rr:>9.4f}{pp:>9.4f}{r2:>9.4f}{p2:>9.4f}')
    print(f'rho range           : {min(loo_rho):.4f} .. {max(loo_rho):.4f}')
    print(f'p range             : {min(loo_p):.4f} .. {max(loo_p):.4f}')
    print(f'sign changes        : {sum(1 for v in loo_rho if v <= 0)} of {len(loo_rho)}')
    print(f'crosses p = 0.05    : {sum(1 for v in loo_p if v > 0.05)} of {len(loo_p)}')

    print()
    print('--- CHECK C: code / spatial margin ---')
    idx = {c: i for i, c in enumerate(categories)}
    for c in ('code', 'spatial'):
        vals = np.array(coh_per_model[c])
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        per_model = '  '.join(f'{m}={v:.4f}' for m, v in zip(MODELS, vals))
        print(f'{c:<8} mean = {coh_mean[idx[c]]:.6f}  SEM = {sem:.6f}')
        print(f'         {per_model}')
    gap = float(coh_mean[idx['code']] - coh_mean[idx['spatial']])
    sem_pooled = float(np.mean([
        np.array(coh_per_model[c]).std(ddof=1) / np.sqrt(len(MODELS))
        for c in ('code', 'spatial')
    ]))
    print(f'gap code - spatial  : {gap:.6e}')
    print(f'mean SEM            : {sem_pooled:.6e}   (ratio {sem_pooled / abs(gap):.1f}x the gap)')

    swapped = coh_mean.copy()
    swapped[idx['code']], swapped[idx['spatial']] = swapped[idx['spatial']], swapped[idx['code']]
    rho_sw, p_sw = spearmanr(swapped, asym_count)
    print(f'rho if the two swap : {rho_sw:.4f}  (parametric p = {p_sw:.4f})')

    print()
    print('--- CALIBRATION: pipeline reimplementation vs committed srez1d ---')
    delta = abs(float(rho_obs) - REFERENCE_RHO)
    verdict = 'MATCH' if delta <= REFERENCE_TOL else 'MISMATCH'
    print(f'this script rho     : {rho_obs:.4f}')
    print(f'srez1d_output.txt   : {REFERENCE_RHO:.4f}')
    print(f'abs difference      : {delta:.6f}  (tolerance {REFERENCE_TOL})')
    print(f'verdict             : {verdict}')
    if verdict == 'MISMATCH':
        raise SystemExit(
            'Pipeline diverged from srez1d_corr.py. Do not cite these numbers '
            'until the divergence is explained.'
        )


if __name__ == '__main__':
    main()
