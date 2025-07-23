"""
Figure 5: RISE Rank Selection Analysis

This figure demonstrates how R²X changes with increasing PARAFAC2 (RISE) rank
to identify the optimal rank where R²X begins to flatline.
"""

import numpy as np
from matplotlib import pyplot as plt
from parafac2.parafac2 import parafac2_nd

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
)
from .common import getSetup, subplotLabel


def makeFigure():
    """Generate Figure 5 showing RISE rank selection analysis."""
    ax, f = getSetup((10, 6), (1, 1))

    print("Importing and preparing data for Figure 5...")
    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")
    print(f"Data shape: {adata_filtered.shape}")

    ranks = list(range(1, 51))
    print(f"Testing ranks 1-50...")

    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    r2x_results = {}
    prev_r2x = None
    marginal_threshold = 0.003
    consecutive_marginal = 0
    max_consecutive = 5

    for i, rank in enumerate(ranks):
        print(f"Testing rank {rank} ({i+1}/{len(ranks)})...")
        try:
            _, r2x = parafac2_nd(
                adata_filtered,
                rank=rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state
            )
            r2x_results[rank] = r2x
            print(f"  Rank {rank}: R²X = {r2x:.6f}")

            # Marginal improvement check
            if prev_r2x is not None:
                improvement = r2x - prev_r2x
                if improvement < marginal_threshold:
                    consecutive_marginal += 1
                    if consecutive_marginal >= max_consecutive:
                        print(f"\nBreaking early: Marginal improvement < {marginal_threshold} for {max_consecutive} consecutive ranks (at rank {rank})")
                        break
                else:
                    consecutive_marginal = 0
            prev_r2x = r2x

        except Exception as e:
            print(f"  Rank {rank}: Failed - {e}")
            r2x_results[rank] = np.nan

    # Filter valid results and plot
    valid_results = {k: v for k, v in r2x_results.items() if not np.isnan(v)}
    ranks = sorted(valid_results.keys())
    r2x_values = [valid_results[r] for r in ranks]

    # Plot R²X curve
    ax[0].plot(ranks, r2x_values, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax[0].set_xlabel('RISE Rank', fontsize=14)
    ax[0].set_ylabel('R²X', fontsize=14)
    ax[0].set_title('PARAFAC2 (RISE) Rank Selection Analysis', fontsize=16)
    ax[0].grid(True, alpha=0.3)

    # Find flatline point (where marginal improvement becomes minimal)
    flatline_rank = find_flatline_rank(ranks, r2x_values, threshold=0.001)
    
    if flatline_rank:
        ax[0].axvline(x=flatline_rank, color='red', linestyle='--', linewidth=2, 
                   label=f'Flatline at Rank {flatline_rank}')
        ax[0].legend()
        print(f"\nRecommendation: R²X begins to flatline around rank {flatline_rank}")
        print(f"R²X at rank {flatline_rank}: {valid_results[flatline_rank]:.6f}")
    
    print("Figure 5 generation complete.")
    return f


def find_flatline_rank(ranks, r2x_values, threshold=0.001):
    """Find the rank where R²X improvement becomes consistently minimal."""
    # Calculate marginal improvements
    improvements = []
    for i in range(1, len(ranks)):
        improvement = r2x_values[i] - r2x_values[i-1]
        improvements.append(improvement)
    
    # Find first rank where 3 consecutive improvements are below threshold
    consecutive_count = 0
    for i, improvement in enumerate(improvements):
        if improvement < threshold:
            consecutive_count += 1
            if consecutive_count >= 3:
                return ranks[i - 1]  # Return the start of the flatline
        else:
            consecutive_count = 0
    
    return None