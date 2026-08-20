"""
Step 1 PCA analysis: last-token confound check
Date: 25 April 2026

This script reproduces all the main slices:
1. Correlation of leading PCs with surface prompt features
2. Category separability vs number of leading PCs (LDA with CV)
3. Poles of each PC (what sits at the ends)
4. R2 from last token on the leading PCs
5. LDA before and after last-token removal
6. Top PCs by category separability on residuals
7. Comparison across all 5 models
8. Layer-by-layer analysis (Qwen only; the other four calls are commented out)

Input: point YADRO_DATA at a directory containing
  - qwen_results.pkl, gemma_results.pkl, olmo_results.pkl,
    falcon_results.pkl, pythia_results.pkl
  - step1_pca/step1_pca_results.pkl

  Requires: pip install numpy scikit-learn scipy

  Run:
    YADRO_DATA=/path/to/yadro_phase2 python analiz_kod.py

Note: step1_pca_results.pkl was produced outside this repository (Step 0,
12 April 2026). Its PCA does not reproduce from the stored activations of
any of the five saved layers; slices reading pca_results are reproducible
from that pickle only. See README.
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Utilities
# ============================================================

def last_token(prompt: str) -> str:
    """Last word of the prompt, punctuation stripped, lowercased."""
    p = prompt.rstrip('.,;:!?()[]{}"\' ')
    words = p.split()
    return words[-1].lower() if words else ""


def surface_features(prompt: str) -> list:
    """9 surface features of the prompt: length, punctuation, markers."""
    pl = prompt.lower()
    return [
        len(prompt),
        len(prompt.split()),
        sum(c.isdigit() for c in prompt),
        sum(c.isupper() for c in prompt),
        sum(c in '.,;:!()[]{}"\'' for c in prompt),
        len(prompt.split()[0]) if prompt.split() else 0,
        int(pl.startswith('if ')),
        int(pl.startswith('the ')),
        int(any(w in pl.split() for w in
                ['one', 'two', 'three', 'four',
                 'first', 'second', 'next'])),
    ]


def residualize(X: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """X after linear removal of controls, column by column."""
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        reg = LinearRegression().fit(controls, X[:, j])
        out[:, j] = X[:, j] - reg.predict(controls)
    return out


# ============================================================
# Loading
# ============================================================

import os
DATA_DIR = Path(os.environ.get('YADRO_DATA', '.'))  # directory holding the *_results.pkl files

with open(DATA_DIR / 'step1_pca' / 'step1_pca_results.pkl', 'rb') as f:
    s1 = pickle.load(f)

prompts = s1['prompts']
categories = np.array(s1['categories'])
models = ['qwen', 'gemma', 'olmo', 'falcon', 'pythia']

last_toks = np.array([last_token(p) for p in prompts])
ohe_tok = OneHotEncoder(sparse_output=False).fit_transform(
    last_toks.reshape(-1, 1))

F_surf = np.array([surface_features(p) for p in prompts], dtype=float)
F_surf_std = StandardScaler().fit_transform(F_surf)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_to_id = {c: i for i, c in enumerate(sorted(set(categories)))}
y_cat = np.array([cat_to_id[c] for c in categories])

print(f"Prompts: {len(prompts)}, models: {len(models)}, "
      f"categories: {len(cat_to_id)}")
print(f"Unique last tokens: {len(set(last_toks))}")
print()


# ============================================================
# Slice 1: PCA vs surface features (Qwen, for illustration)
# ============================================================

def srez_surface_correlation(model='qwen'):
    print("=" * 72)
    print(f"SLICE: PCA vs surface features ({model})")
    print("=" * 72)

    X_pca = s1['pca_results'][model]['X_pca']
    evr = s1['pca_results'][model]['explained_variance_ratio']

    print(f"\n{'PC':<5}{'evr%':<8}{'R²(surface)':<12}")
    print("-" * 28)
    weighted = 0
    total = 0
    for pc in range(10):
        r2 = LinearRegression().fit(F_surf, X_pca[:, pc]).score(
            F_surf, X_pca[:, pc])
        weighted += r2 * evr[pc]
        total += evr[pc]
        print(f"PC{pc+1:<3} {evr[pc]*100:<7.2f} {r2:<11.3f}")
    print(f"\nShare of variance in the first 10 PCs explained by surface: "
          f"{weighted/total*100:.1f}%")
    print()


# ============================================================
# Slice 2: category separability vs number of leading PCs (Qwen)
# ============================================================

def srez_dim_vs_lda(model='qwen'):
    print("=" * 72)
    print(f"SLICE: category separability vs number of leading PCs ({model})")
    print("=" * 72)

    X_pca = s1['pca_results'][model]['X_pca']
    evr = s1['pca_results'][model]['explained_variance_ratio']

    print(f"\n{'dim':>4} {'sil':<7} {'LDA CV':<8} {'cum.var':<8}")
    for k in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40]:
        Xk = X_pca[:, :k]
        sil = silhouette_score(Xk, y_cat) if k >= 2 else float('nan')
        try:
            acc = cross_val_score(LinearDiscriminantAnalysis(),
                                  Xk, y_cat, cv=cv).mean()
        except Exception:
            acc = float('nan')
        cum = evr[:k].sum()
        sil_s = f"{sil:.3f}" if not np.isnan(sil) else "  -- "
        print(f"{k:>4} {sil_s:<7} {acc:.3f}    {cum*100:5.1f}%")

    # Baselines
    print("\nBaseline:")
    print(f"  Random:       {1/len(cat_to_id):.3f}")
    acc_tok = cross_val_score(LinearDiscriminantAnalysis(),
                              ohe_tok, y_cat, cv=cv).mean()
    acc_surf = cross_val_score(LinearDiscriminantAnalysis(),
                               F_surf_std, y_cat, cv=cv).mean()
    print(f"  By last token:    {acc_tok:.3f}")
    print(f"  By surface feats: {acc_surf:.3f}")
    print()


# ============================================================
# Slice 3: poles of each PC (40 PCs, 5 prompts per pole)
# ============================================================

def srez_pc_extremes(model='qwen', save_to=None):
    X_pca = s1['pca_results'][model]['X_pca']
    evr = s1['pca_results'][model]['explained_variance_ratio']

    out = []
    out.append("=" * 90)
    out.append(f"SLICE: poles of each PC ({model}, all 40 PCs)")
    out.append("=" * 90)
    out.append("")
    out.append("For each PC: the 5 prompts with the highest projection (+) "
               "and the 5 with the lowest (-).")
    out.append("In brackets: prompt category and projection value.")
    out.append("")

    for pc in range(40):
        proj = X_pca[:, pc]
        order = np.argsort(proj)
        top = order[-5:][::-1]
        bottom = order[:5]
        out.append(f"━━━ PC{pc+1} (explains {evr[pc]*100:.2f}%) ━━━")
        out.append("")
        out.append(f"  + pole:")
        for idx in top:
            out.append(f"    [{categories[idx]:>9}] ({proj[idx]:+.2f})  "
                       f"{prompts[idx]}")
        out.append(f"  − pole:")
        for idx in bottom:
            out.append(f"    [{categories[idx]:>9}] ({proj[idx]:+.2f})  "
                       f"{prompts[idx]}")
        out.append("")

    text = "\n".join(out)
    if save_to:
        Path(save_to).write_text(text)
        print(f"Saved: {save_to}")
    else:
        print(text[:3000])
        print("...")
    return text


# ============================================================
# Slices 4 + 5: last-token confound across all 5 models
# ============================================================

def srez_last_token_all_models():
    print("=" * 72)
    print("SLICE: last-token confound across all 5 models")
    print("=" * 72)

    print("\nR² from last token for the top-7 PCs:")
    print(f"{'model':<8}", " ".join(f"  PC{i+1}" for i in range(7)))
    for m in models:
        X = s1['pca_results'][m]['X_pca']
        r2s = []
        for pc in range(7):
            r2 = LinearRegression().fit(ohe_tok, X[:, pc]).score(
                ohe_tok, X[:, pc])
            r2s.append(r2)
        print(f"{m:<8}", " ".join(f"{r:.2f}" for r in r2s))

    print("\nLDA before and after last-token removal:")
    print(f"{'model':<8}{'orig 7':<9}{'resid 7':<10}{'loss':<10}"
          f"{'orig 15':<10}{'resid 15':<11}{'loss':<10}")

    results = {}
    for m in models:
        X = s1['pca_results'][m]['X_pca']
        X_resid = residualize(X, ohe_tok)

        a7 = cross_val_score(LinearDiscriminantAnalysis(),
                             X[:, :7], y_cat, cv=cv).mean()
        a7r = cross_val_score(LinearDiscriminantAnalysis(),
                              X_resid[:, :7], y_cat, cv=cv).mean()
        a15 = cross_val_score(LinearDiscriminantAnalysis(),
                              X[:, :15], y_cat, cv=cv).mean()
        a15r = cross_val_score(LinearDiscriminantAnalysis(),
                               X_resid[:, :15], y_cat, cv=cv).mean()

        print(f"{m:<8}{a7:<8.3f} {a7r:<9.3f} {(a7-a7r)*100:<9.1f} "
              f"{a15:<9.3f} {a15r:<10.3f} {(a15-a15r)*100:<9.1f}")
        results[m] = {'X': X, 'X_resid': X_resid}
    print()
    return results


# ============================================================
# Slice 6: top PCs by category separability on residuals
# ============================================================

def srez_top_resid_pc(results):
    print("=" * 72)
    print("Top-3 PCs by 1D LDA on residuals (where clean semantics sits)")
    print("=" * 72)

    top = {}
    for m in models:
        X_resid = results[m]['X_resid']
        accs = []
        for pc in range(40):
            try:
                a = cross_val_score(LinearDiscriminantAnalysis(),
                                    X_resid[:, pc:pc+1], y_cat, cv=cv).mean()
            except Exception:
                a = 0.0
            accs.append((pc, a))
        accs.sort(key=lambda x: x[1], reverse=True)
        top[m] = accs[:3]
        evr = s1['pca_results'][m]['explained_variance_ratio']
        print(f"\n{m.upper()}:")
        for rank, (pc, a) in enumerate(accs[:3]):
            print(f"  {rank+1}. PC{pc+1} 1D LDA={a:.3f}, "
                  f"orig evr={evr[pc]*100:.2f}%")

    # Poles of the top-1 PC per model
    print("\n" + "=" * 72)
    print("Poles of the top-1 PC (on residuals) per model - categories")
    print("=" * 72)
    for m in models:
        pc, acc = top[m][0]
        evr = s1['pca_results'][m]['explained_variance_ratio']
        proj = results[m]['X_resid'][:, pc]
        order = np.argsort(proj)
        top_cats = [categories[i] for i in order[-5:]]
        bot_cats = [categories[i] for i in order[:5]]
        print(f"\n{m.upper()}: PC{pc+1} (orig {evr[pc]*100:.2f}%, "
              f"acc={acc:.3f})")
        print(f"  + : " + ", ".join(top_cats))
        print(f"  − : " + ", ".join(bot_cats))
    print()
    return top


# ============================================================
# Slice 7: cross-model invariance of the top PCs
# ============================================================

def srez_invariance(results, top):
    print("=" * 72)
    print("Cross-model invariance of top PCs (on residuals)")
    print("=" * 72)

    print("\n|Pearson r| between top-1 PCs of two models:")
    print(f"{'':<10}" + " ".join(f"{m:<7}" for m in models))
    for m1 in models:
        pc1 = top[m1][0][0]
        proj1 = results[m1]['X_resid'][:, pc1]
        row = []
        for m2 in models:
            pc2 = top[m2][0][0]
            proj2 = results[m2]['X_resid'][:, pc2]
            r = abs(np.corrcoef(proj1, proj2)[0, 1])
            row.append(f"{r:.3f}")
        print(f"{m1:<10}" + " ".join(f"{v:<7}" for v in row))

    print("\nMax |r| between a top-1 PC and any PC of the other model:")
    print(f"{'':<10}" + " ".join(f"{m:<7}" for m in models))
    for m1 in models:
        pc1 = top[m1][0][0]
        proj1 = results[m1]['X_resid'][:, pc1]
        row = []
        for m2 in models:
            if m1 == m2:
                row.append("1.000")
                continue
            mx = 0
            for pc2 in range(40):
                proj2 = results[m2]['X_resid'][:, pc2]
                r = abs(np.corrcoef(proj1, proj2)[0, 1])
                if r > mx:
                    mx = r
            row.append(f"{mx:.3f}")
        print(f"{m1:<10}" + " ".join(f"{v:<7}" for v in row))
    print()


# ============================================================
# Slice 8: layer-by-layer analysis (requires *_results.pkl with activations)
# ============================================================

def srez_by_layer(model='qwen'):
    """Requires <model>_results.pkl with per-layer activations."""
    pkl_path = DATA_DIR / f'{model}_results.pkl'
    if not pkl_path.exists():
        print(f"No {pkl_path} - skipping the layer slice")
        return

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    print("=" * 72)
    print(f"LAYER-BY-LAYER ANALYSIS: {model}")
    print("=" * 72)

    layers = sorted(raw['activations'].keys())
    print(f"\n{'layer':<8}{'PC1 evr':<10}{'avg R²(tok)':<14}"
          f"{'orig 7':<9}{'resid 7':<10}{'orig 15':<10}{'resid 15':<11}")
    print("-" * 70)

    for layer in layers:
        X_raw = raw['activations'][layer]
        pca = PCA(n_components=40, svd_solver='full')
        X_pca = pca.fit_transform(X_raw)
        evr = pca.explained_variance_ratio_

        r2s = []
        for pc in range(7):
            r2 = LinearRegression().fit(ohe_tok, X_pca[:, pc]).score(
                ohe_tok, X_pca[:, pc])
            r2s.append(r2)

        X_resid = residualize(X_pca, ohe_tok)
        a7 = cross_val_score(LinearDiscriminantAnalysis(),
                             X_pca[:, :7], y_cat, cv=cv).mean()
        a7r = cross_val_score(LinearDiscriminantAnalysis(),
                              X_resid[:, :7], y_cat, cv=cv).mean()
        a15 = cross_val_score(LinearDiscriminantAnalysis(),
                              X_pca[:, :15], y_cat, cv=cv).mean()
        a15r = cross_val_score(LinearDiscriminantAnalysis(),
                               X_resid[:, :15], y_cat, cv=cv).mean()

        print(f"{layer:<8}{evr[0]*100:<9.2f} {np.mean(r2s):<13.3f} "
              f"{a7:<8.3f} {a7r:<9.3f} {a15:<9.3f} {a15r:<10.3f}")
    print()


# ============================================================
# Run all slices
# ============================================================

if __name__ == '__main__':
    srez_surface_correlation('qwen')
    srez_dim_vs_lda('qwen')
    srez_pc_extremes('qwen', save_to='srez3_pc_extremes_qwen.txt')
    srez_pc_extremes('gemma', save_to='srez3_pc_extremes_gemma.txt')
    results = srez_last_token_all_models()
    top = srez_top_resid_pc(results)
    srez_invariance(results, top)
    srez_by_layer('qwen')
    # The other _results.pkl files are present; uncomment to include them:
    # srez_by_layer('gemma')
    # srez_by_layer('olmo')
    # srez_by_layer('falcon')
    # srez_by_layer('pythia')
