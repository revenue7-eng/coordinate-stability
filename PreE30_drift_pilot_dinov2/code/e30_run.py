"""
E30: Coordinate Drift on Production-Scale Self-Supervised Vision Models
=======================================================================
Compares DINOv2-small (22M, 384D) and DINOv2-base (86M, 768D) embeddings
on the same images. Tests whether different model capacities produce
alignable coordinate systems despite identical training data and procedure.

Hypothesis (Andrew's drift theory):
  If free latent space loses coordinate identifiability, then the same
  training data + same algorithm + different capacity should give NON-alignable
  coordinate systems (low Procrustes R² after dimension equalization).

Three metrics:
  - Procrustes R² (raw, Andrew's E27 formula)
  - Procrustes R² (Frobenius-normalized, scale-invariant)
  - Linear CKA (geometry similarity, rotation-invariant)

If R² is low but CKA is high → coordinate drift without geometry collapse
(Andrew's central thesis confirmed at production scale).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from transformers import AutoImageProcessor, AutoModel
from torchvision import datasets, transforms

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("../data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = 500          # how many images to use
SEED = 42
DEVICE = "cpu"
BATCH_SIZE = 4          # small for CPU memory safety

MODELS = [
    ("dinov2-small", "facebook/dinov2-small", 384),
    ("dinov2-base",  "facebook/dinov2-base",  768),
]

print(f"E30 starting | n_images={N_IMAGES} | device={DEVICE}")
print("=" * 70)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. LOAD DATASET (CIFAR-100 test split, subset)
# ============================================================
print("\n[1/5] Loading CIFAR-100 test split...")
t0 = time.time()

# Just download, no processor transforms here — we'll preprocess per-model
cifar = datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True)

# Random subset of N_IMAGES
rng = np.random.RandomState(SEED)
indices = rng.choice(len(cifar), size=N_IMAGES, replace=False)
indices = sorted(indices.tolist())  # sorted for reproducibility

# Keep raw PIL images + labels (for later analysis if needed)
images_raw = []
labels = []
for idx in indices:
    img, lbl = cifar[idx]   # img is PIL Image (RGB, 32x32)
    images_raw.append(img.convert("RGB"))
    labels.append(int(lbl))

print(f"  loaded {len(images_raw)} images in {time.time() - t0:.1f}s")


# ============================================================
# 2. EXTRACT EMBEDDINGS PER MODEL
# ============================================================
def extract_embeddings(model_name, hf_id, expected_dim):
    """Run model on all images, return (N, D) array of CLS embeddings."""
    print(f"\n[2/5] Loading {model_name} ({hf_id})...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    embs = []
    t0 = time.time()
    print(f"  running inference on {N_IMAGES} images (batch={BATCH_SIZE})...")
    with torch.no_grad():
        for i in range(0, N_IMAGES, BATCH_SIZE):
            batch_imgs = images_raw[i:i + BATCH_SIZE]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            # CLS token = first token of last_hidden_state, OR pooler_output
            cls = outputs.last_hidden_state[:, 0, :]   # (B, D)
            embs.append(cls.cpu().numpy())
            if (i // BATCH_SIZE) % 20 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (N_IMAGES - i) / rate
                print(f"    {i}/{N_IMAGES}  rate={rate:.1f} img/s  ETA={eta:.0f}s")

    embs = np.concatenate(embs, axis=0)
    assert embs.shape == (N_IMAGES, expected_dim), \
        f"expected ({N_IMAGES},{expected_dim}), got {embs.shape}"
    print(f"  done in {time.time() - t0:.0f}s | shape={embs.shape}")

    # Free memory before loading next model
    del model
    del processor
    import gc
    gc.collect()

    return embs


emb_small = extract_embeddings(*MODELS[0])
emb_base = extract_embeddings(*MODELS[1])

# Save raw embeddings for reproducibility
np.savez(RESULTS_DIR / "embeddings.npz",
         emb_small=emb_small, emb_base=emb_base,
         labels=np.array(labels), indices=np.array(indices))
print(f"\n  saved embeddings to {RESULTS_DIR / 'embeddings.npz'}")


# ============================================================
# 3. EQUALIZE DIMENSIONS (PCA: base 768D → 384D)
# ============================================================
print("\n[3/5] Equalizing dimensions via PCA...")

def pca_reduce(X, target_dim):
    """Center, then SVD, return top-k components."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Project onto top target_dim components
    return Xc @ Vt[:target_dim].T, S

# Reduce base from 768D to 384D
emb_base_pca, base_singular = pca_reduce(emb_base, 384)
# small is already 384D, but center+rotate to its own PCA basis for fair comparison
emb_small_pca, small_singular = pca_reduce(emb_small, 384)

print(f"  small: rank-effective check, top-5 singular values: {small_singular[:5]}")
print(f"  base:  rank-effective check, top-5 singular values: {base_singular[:5]}")
print(f"  small explained variance (top 384/{emb_small.shape[1]}): "
      f"{(small_singular[:384]**2).sum() / (small_singular**2).sum():.4f}")
print(f"  base  explained variance (top 384/{emb_base.shape[1]}):  "
      f"{(base_singular[:384]**2).sum() / (base_singular**2).sum():.4f}")


# ============================================================
# 4. METRICS
# ============================================================
print("\n[4/5] Computing similarity metrics...")

def andrew_r2_raw(emb_ref, emb_curr):
    """E27 formula: Procrustes R² without scale normalization."""
    ref_c = emb_ref - emb_ref.mean(0)
    curr_c = emb_curr - emb_curr.mean(0)
    R, _ = orthogonal_procrustes(curr_c, ref_c)
    aligned = curr_c @ R
    residual = np.mean((aligned - ref_c) ** 2)
    total_var = np.mean(ref_c ** 2)
    return 1.0 - residual / total_var if total_var > 0 else 0.0


def andrew_r2_frob(emb_ref, emb_curr):
    """E27 formula + Frobenius normalization (scale-invariant)."""
    ref_c = emb_ref - emb_ref.mean(0)
    curr_c = emb_curr - emb_curr.mean(0)
    ref_n = ref_c / np.linalg.norm(ref_c)
    curr_n = curr_c / np.linalg.norm(curr_c)
    R, _ = orthogonal_procrustes(curr_n, ref_n)
    aligned = curr_n @ R
    residual = np.sum((aligned - ref_n) ** 2)
    total_var = np.sum(ref_n ** 2)
    return 1.0 - residual / total_var


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    num = np.sum(XtY ** 2)
    den = np.sqrt(np.sum(XtX ** 2) * np.sum(YtY ** 2))
    return float(num / den) if den > 0 else 0.0


# Compute on PCA-reduced versions (both 384D)
r2_raw  = andrew_r2_raw(emb_small_pca, emb_base_pca)
r2_frob = andrew_r2_frob(emb_small_pca, emb_base_pca)
cka     = linear_cka(emb_small_pca, emb_base_pca)

# Also CKA on full original dimensions (CKA handles different dims fine)
cka_full = linear_cka(emb_small, emb_base)

# Sanity: model self-comparison on shuffled subset
half = N_IMAGES // 2
r2_self_small = andrew_r2_raw(emb_small_pca[:half], emb_small_pca[:half])
cka_random = linear_cka(emb_small_pca, np.random.randn(*emb_small_pca.shape))


# ============================================================
# 5. REPORT
# ============================================================
print("\n[5/5] Results")
print("=" * 70)
print(f"\nDINOv2-small vs DINOv2-base (after PCA → 384D for both)")
print(f"  N images:            {N_IMAGES}")
print(f"  small dim:           384  (native)")
print(f"  base  dim:           768 → 384 (PCA)")
print()
print(f"  Procrustes R² (raw):       {r2_raw:>8.4f}")
print(f"  Procrustes R² (Frob-norm): {r2_frob:>8.4f}")
print(f"  Linear CKA (PCA-reduced):  {cka:>8.4f}")
print(f"  Linear CKA (full dim):     {cka_full:>8.4f}")
print()
print(f"  Sanity checks:")
print(f"    R² small-vs-self (should ≈ 1):     {r2_self_small:>8.4f}")
print(f"    CKA vs random noise (should ≈ 0):  {cka_random:>8.4f}")
print()
print("=" * 70)
print("INTERPRETATION:")
print()
if cka > 0.7 and r2_frob < 0.5:
    print("  → HIGH CKA + LOW R² = geometry preserved, coordinates drift.")
    print("    This SUPPORTS Andrew's drift hypothesis at production scale.")
elif cka > 0.7 and r2_frob > 0.7:
    print("  → HIGH CKA + HIGH R² = stable coordinates and geometry.")
    print("    Drift NOT detected at this scale. Theory may be small-scale.")
elif cka < 0.3:
    print("  → LOW CKA = models genuinely encode different content.")
    print("    Beyond drift — fundamental representation difference.")
else:
    print("  → Mixed signal. Need careful interpretation.")
print()

# Save results
results = {
    "config": {
        "n_images": N_IMAGES,
        "seed": SEED,
        "models": [m[0] for m in MODELS],
        "dataset": "CIFAR-100 test split (random subset)",
    },
    "metrics": {
        "procrustes_r2_raw": float(r2_raw),
        "procrustes_r2_frob_norm": float(r2_frob),
        "linear_cka_pca_reduced": float(cka),
        "linear_cka_full_dim": float(cka_full),
    },
    "sanity": {
        "r2_self_small": float(r2_self_small),
        "cka_vs_random": float(cka_random),
    },
    "explained_variance": {
        "small_top384_of_384": float((small_singular[:384]**2).sum() / (small_singular**2).sum()),
        "base_top384_of_768": float((base_singular[:384]**2).sum() / (base_singular**2).sum()),
    },
}

with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {RESULTS_DIR / 'results.json'}")
print(f"  Embeddings saved to {RESULTS_DIR / 'embeddings.npz'}")
print()
print("Done.")
