"""
Figure 5: RISE Rank Selection Analysis

This figure demonstrates how R²X changes with increasing PARAFAC2 (RISE) rank
to identify the optimal rank where R²X begins to flatline.
"""

import numpy as np
from parafac2.parafac2 import parafac2_nd

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
)
from .common import getSetup


def makeFigure():
    """Generate Figure 5 showing RISE rank selection analysis."""
    ax, f = getSetup((6, 3), (1, 1))

    print("Importing and preparing data for Figure 5...")
    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")
    print(f"Data shape: {adata_filtered.shape}")

    ranks = list(range(1, 51, 5))
    print(f"Testing ranks {ranks[0]}-{ranks[-1]} in steps of 5...")

    n_iter_max = 1000
    tol = 1e-8
    random_state = 42

    r2x_results = {}

    for i, rank in enumerate(ranks):
        print(f"Testing rank {rank} ({i+1}/{len(ranks)})...")
        try:
            _, r2x = parafac2_nd(
                adata_filtered,
                rank=rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
            )
            r2x_results[rank] = r2x
            print(f"  Rank {rank}: R²X = {r2x:.6f}")

        except Exception as e:
            print(f"  Rank {rank}: Failed - {e}")
            r2x_results[rank] = np.nan

    # Filter valid results and plot
    valid_results = {k: v for k, v in r2x_results.items() if not np.isnan(v)}
    ranks = sorted(valid_results.keys())
    r2x_values = [valid_results[r] for r in ranks]

    # Plot R²X curve
    ax[0].plot(ranks, r2x_values, "o-", color="steelblue")
    ax[0].set_xlabel("RISE Rank")
    ax[0].set_ylabel("R²X")
    ax[0].set_title("PARAFAC2 (RISE) Rank Selection Analysis")
    # ax[0].grid(True, alpha=0.3)

    # Set y min and max to 0 - 1.0
    ax[0].set_ylim(0, 0.2)

    print("Figure 5 generation complete.")
    return f
