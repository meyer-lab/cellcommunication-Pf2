"""PARAFAC2 Rank Selection via Cell Holdout Cross-Validation."""

import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import issparse

from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from tensorly.parafac2_tensor import parafac2_to_slices

try:
    from parafac2.parafac2 import parafac2_nd
except ImportError:
    raise ImportError("Could not import 'parafac2_nd'.")

jax.config.update("jax_enable_x64", True)


def create_stratified_splits(adata, condition_key, n_folds=5, random_state=42):
    """Create K-Fold splits stratified by condition."""
    if condition_key not in adata.obs:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    conditions = adata.obs[condition_key].astype('category').cat.codes.values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    return [(i, train_idx, test_idx) 
            for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(conditions)), conditions))]


def reconstruct_from_pf2(pf2_output, condition_idxs):
    """Reconstruct expression matrix from PARAFAC2 output."""
    weights, factors, projections = pf2_output
    slices = parafac2_to_slices((weights, factors, projections))
    
    n_genes = factors[2].shape[0]
    n_cells = len(condition_idxs)
    X_recon = np.zeros((n_cells, n_genes), dtype=np.float64)
    
    for cond_idx in np.unique(condition_idxs):
        mask = (condition_idxs == cond_idx)
        X_recon[mask, :] = slices[cond_idx]
        
    return X_recon


def compute_ot_score(X_recon, X_real, epsilon=0.1):
    """Compute Sinkhorn divergence between reconstructed and real data."""
    if issparse(X_real):
        X_real = X_real.toarray()
    
    X_real = np.asarray(X_real, dtype=np.float64)
    X_recon = np.asarray(X_recon, dtype=np.float64)
    
    scale = np.mean(np.linalg.norm(X_real, axis=1))
    if scale == 0:
        scale = 1.0
    
    X_real_norm = jnp.array(X_real / scale, dtype=jnp.float64)
    X_recon_norm = jnp.array(X_recon / scale, dtype=jnp.float64)
    
    div = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x=X_recon_norm, y=X_real_norm, epsilon=epsilon
    )
    
    return float(div[0]) if isinstance(div, tuple) else float(div.divergence)


def run_rank_selection(
    adata, ranks, condition_key,
    n_folds=5, n_iter_max=100, tol=1e-6, ot_epsilon=0.1, random_state=1
):
    """Run cross-validation pipeline for rank selection.
    
    Returns DataFrame with columns: rank, ot_score_mean, ot_score_std, r2x_mean
    """
    from ..import_data import add_cond_idxs
    
    print(f"Starting Rank Selection ({n_folds}-fold CV)...")
    print(f"Testing ranks: {ranks}")
    
    splits = create_stratified_splits(adata, condition_key, n_folds, random_state)
    results = []

    for r in ranks:
        print(f"\n--- Rank {r} ---")
        fold_scores = []
        fold_r2x = []
        start_time = time.time()
        
        for fold_idx, train_idx, test_idx in splits:
            train_adata = add_cond_idxs(adata[train_idx].copy(), condition_key)
            test_adata = adata[test_idx]
            
            try:
                pf2_output, r2x = parafac2_nd(
                    train_adata, rank=r, n_iter_max=n_iter_max, 
                    tol=tol, random_state=random_state
                )
            except Exception as e:
                print(f"Fit failed for Rank {r}: {e}")
                continue

            train_cond_idxs = train_adata.obs["condition_unique_idxs"].values
            X_train_recon = reconstruct_from_pf2(pf2_output, train_cond_idxs)
            
            X_test_real = test_adata.X
            score = compute_ot_score(X_train_recon, X_test_real, ot_epsilon)
            
            fold_scores.append(score)
            fold_r2x.append(r2x)
            print(f"   Fold {fold_idx+1}: OT = {score:.4f}")

        if fold_scores:
            results.append({
                "rank": r,
                "ot_score_mean": np.mean(fold_scores),
                "ot_score_std": np.std(fold_scores),
                "r2x_mean": np.mean(fold_r2x)
            })
            print(f"Rank {r} done in {time.time() - start_time:.1f}s | Mean OT: {np.mean(fold_scores):.4f}")

    return pd.DataFrame(results)
