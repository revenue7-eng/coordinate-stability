#!/usr/bin/env python3
"""
Drift Rate Correlation Analysis
=================================
Computes Pearson and Spearman correlation between raw_drift and val_loss
across epochs, using data already saved in full_results.json from
random_fixed_full.py.

Also computes drift between consecutive embedding snapshots and
correlates with val_loss trajectory.

Run:
  py -3.12 drift_rate_correlation.py --results-dir random_fixed_full_results

Author: Andrew + Claude (prescribed axes project)
"""

import json, argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import orthogonal_procrustes


def load_embeddings(results_dir, mode, seed, epoch):
    """Load saved embeddings from .npz file."""
    path = results_dir / f"{mode}_seed{seed}_emb_ep{epoch}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["emb"]


def compute_drift(emb_ref, emb_curr):
    """Compute raw drift and Procrustes-aligned drift between two embedding sets."""
    # Raw drift (no alignment)
    raw = np.mean((emb_curr - emb_ref)**2)

    # Center
    ref_c = emb_ref - emb_ref.mean(0)
    curr_c = emb_curr - emb_curr.mean(0)

    # Procrustes alignment
    R, _ = orthogonal_procrustes(curr_c, ref_c)
    aligned = curr_c @ R

    residual = np.mean((aligned - ref_c)**2)
    total_var = np.mean(ref_c**2)
    r_squared = 1.0 - residual / total_var if total_var > 0 else 0.0

    return {
        "raw_drift": float(raw),
        "procrustes_residual": float(residual),
        "procrustes_r_squared": float(r_squared),
        "rotation_det": float(np.linalg.det(R))
    }


def get_val_loss_at_epoch(history_path, epoch):
    """Get val loss at specific epoch from history file."""
    with open(history_path) as f:
        hist = json.load(f)
    for entry in hist:
        if entry["ep"] == epoch:
            return entry["vp"]
    return None


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--results-dir", type=str, default="random_fixed_full_results",
                    help="Directory with results from random_fixed_full.py")
    args = pa.parse_args()

    results_dir = Path(args.results_dir)

    # Load full results to get metadata
    full_results_path = results_dir / "full_results.json"
    if not full_results_path.exists():
        print(f"ERROR: {full_results_path} not found.")
        print("Run random_fixed_full.py first to generate data.")
        return

    with open(full_results_path) as f:
        full_results = json.load(f)

    seeds = full_results["training_seeds"]
    save_epochs = full_results["save_epochs"]
    modes = ["prescribed", "random_fixed", "free"]

    print(f"Seeds: {seeds}")
    print(f"Save epochs: {save_epochs}")
    print(f"Modes: {modes}")

    # ==========================================
    #  ANALYSIS 1: Cumulative drift from epoch 1
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 1: CUMULATIVE DRIFT FROM EPOCH 1")
    print(f"{'='*60}")

    all_correlations = {}

    for mode in modes:
        for seed in seeds:
            key = f"{mode}_seed{seed}"
            ref_emb = load_embeddings(results_dir, mode, seed, save_epochs[0])
            if ref_emb is None:
                print(f"  {key}: no embeddings found, skipping")
                continue

            history_path = results_dir / f"{mode}_seed{seed}_history.json"
            if not history_path.exists():
                print(f"  {key}: no history found, skipping")
                continue

            drifts = []
            val_losses = []
            epochs_used = []

            for ep in save_epochs:
                curr_emb = load_embeddings(results_dir, mode, seed, ep)
                if curr_emb is None:
                    continue
                drift_data = compute_drift(ref_emb, curr_emb)
                vl = get_val_loss_at_epoch(history_path, ep)
                if vl is None:
                    continue

                drifts.append(drift_data["raw_drift"])
                val_losses.append(vl)
                epochs_used.append(ep)

            print(f"\n  --- {key} ---")
            print(f"  {'Epoch':>6} | {'Raw Drift':>12} | {'Val Loss':>12}")
            print(f"  {'-'*36}")
            for i, ep in enumerate(epochs_used):
                print(f"  {ep:6d} | {drifts[i]:12.6f} | {val_losses[i]:12.6f}")

            if len(drifts) >= 3:
                pearson_r, pearson_p = pearsonr(drifts, val_losses)
                spearman_r, spearman_p = spearmanr(drifts, val_losses)
                print(f"\n  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
                print(f"  Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

                all_correlations[key] = {
                    "epochs": epochs_used,
                    "raw_drifts": drifts,
                    "val_losses": val_losses,
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p)
                }
            else:
                print(f"\n  Not enough data points for correlation (need ≥3, have {len(drifts)})")

    # ==========================================
    #  ANALYSIS 2: Consecutive drift
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 2: CONSECUTIVE DRIFT (EPOCH-TO-EPOCH)")
    print(f"{'='*60}")

    consecutive_correlations = {}

    for mode in modes:
        for seed in seeds:
            key = f"{mode}_seed{seed}"
            history_path = results_dir / f"{mode}_seed{seed}_history.json"
            if not history_path.exists():
                continue

            consec_drifts = []
            consec_losses = []
            consec_epochs = []

            for i in range(1, len(save_epochs)):
                ep_prev = save_epochs[i-1]
                ep_curr = save_epochs[i]
                emb_prev = load_embeddings(results_dir, mode, seed, ep_prev)
                emb_curr = load_embeddings(results_dir, mode, seed, ep_curr)
                if emb_prev is None or emb_curr is None:
                    continue

                drift_data = compute_drift(emb_prev, emb_curr)
                vl = get_val_loss_at_epoch(history_path, ep_curr)
                if vl is None:
                    continue

                consec_drifts.append(drift_data["raw_drift"])
                consec_losses.append(vl)
                consec_epochs.append(f"{ep_prev}->{ep_curr}")

            print(f"\n  --- {key} ---")
            print(f"  {'Interval':>10} | {'Drift':>12} | {'Val Loss':>12}")
            print(f"  {'-'*40}")
            for i, ep_str in enumerate(consec_epochs):
                print(f"  {ep_str:>10} | {consec_drifts[i]:12.6f} | {consec_losses[i]:12.6f}")

            if len(consec_drifts) >= 3:
                pearson_r, pearson_p = pearsonr(consec_drifts, consec_losses)
                spearman_r, spearman_p = spearmanr(consec_drifts, consec_losses)
                print(f"\n  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
                print(f"  Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

                consecutive_correlations[key] = {
                    "intervals": consec_epochs,
                    "drifts": consec_drifts,
                    "val_losses": consec_losses,
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p)
                }

    # ==========================================
    #  ANALYSIS 3: Cross-condition comparison
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 3: CROSS-CONDITION (DRIFT vs LOSS AT FINAL EPOCH)")
    print(f"{'='*60}")

    final_ep = save_epochs[-1]
    cross_drifts = []
    cross_losses = []
    cross_labels = []

    for mode in modes:
        for seed in seeds:
            ref_emb = load_embeddings(results_dir, mode, seed, save_epochs[0])
            final_emb = load_embeddings(results_dir, mode, seed, final_ep)
            if ref_emb is None or final_emb is None:
                continue

            history_path = results_dir / f"{mode}_seed{seed}_history.json"
            vl = get_val_loss_at_epoch(history_path, final_ep)
            if vl is None:
                continue

            drift_data = compute_drift(ref_emb, final_emb)
            cross_drifts.append(drift_data["raw_drift"])
            cross_losses.append(vl)
            cross_labels.append(f"{mode}_s{seed}")

    print(f"\n  {'Condition':>20} | {'Drift (ep1→{})'.format(final_ep):>16} | {'Val Loss':>12}")
    print(f"  {'-'*55}")
    for i, label in enumerate(cross_labels):
        print(f"  {label:>20} | {cross_drifts[i]:16.6f} | {cross_losses[i]:12.6f}")

    if len(cross_drifts) >= 3:
        pearson_r, pearson_p = pearsonr(cross_drifts, cross_losses)
        spearman_r, spearman_p = spearmanr(cross_drifts, cross_losses)
        print(f"\n  Cross-condition Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"  Cross-condition Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

    # ==========================================
    #  SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Cumulative drift correlations (drift from ep1):")
    for key, data in all_correlations.items():
        print(f"    {key:>25}: Pearson r={data['pearson_r']:.4f} (p={data['pearson_p']:.4f}), Spearman r={data['spearman_r']:.4f} (p={data['spearman_p']:.4f})")

    print(f"\n  Consecutive drift correlations (epoch-to-epoch):")
    for key, data in consecutive_correlations.items():
        print(f"    {key:>25}: Pearson r={data['pearson_r']:.4f} (p={data['pearson_p']:.4f}), Spearman r={data['spearman_r']:.4f} (p={data['spearman_p']:.4f})")

    if len(cross_drifts) >= 3:
        print(f"\n  Cross-condition (all modes, all seeds, final epoch):")
        print(f"    Pearson r={pearson_r:.4f} (p={pearson_p:.4f}), Spearman r={spearman_r:.4f} (p={spearman_p:.4f})")

    print(f"\n  Interpretation:")
    # Check free correlations
    free_corrs = {k: v for k, v in all_correlations.items() if "free" in k and "random" not in k}
    if free_corrs:
        avg_r = np.mean([v["pearson_r"] for v in free_corrs.values()])
        if avg_r > 0.8:
            print(f"    Within free encoder: strong positive correlation (avg Pearson r={avg_r:.3f})")
            print(f"    → Drift and loss decrease together over training")
            print(f"    → This is EXPECTED (both improve with training) and does NOT prove causation")
        elif avg_r < -0.5:
            print(f"    Within free encoder: negative correlation (avg Pearson r={avg_r:.3f})")
            print(f"    → More drift = lower loss (counterintuitive)")
        else:
            print(f"    Within free encoder: weak correlation (avg Pearson r={avg_r:.3f})")

    if len(cross_drifts) >= 3:
        if pearson_r > 0.8:
            print(f"    Cross-condition: strong positive correlation (r={pearson_r:.3f})")
            print(f"    → Conditions with more drift have higher loss")
            print(f"    → Consistent with drift→loss hypothesis, but does not prove causation")
            print(f"    → (prescribed has zero drift AND lowest loss; free has drift AND highest loss)")
        else:
            print(f"    Cross-condition: correlation r={pearson_r:.3f}")

    # Save
    output = {
        "cumulative_drift_correlations": all_correlations,
        "consecutive_drift_correlations": consecutive_correlations,
        "cross_condition": {
            "labels": cross_labels,
            "drifts": cross_drifts,
            "val_losses": cross_losses,
            "pearson_r": float(pearson_r) if len(cross_drifts) >= 3 else None,
            "pearson_p": float(pearson_p) if len(cross_drifts) >= 3 else None,
            "spearman_r": float(spearman_r) if len(cross_drifts) >= 3 else None,
            "spearman_p": float(spearman_p) if len(cross_drifts) >= 3 else None,
        }
    }

    out_path = results_dir / "drift_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
