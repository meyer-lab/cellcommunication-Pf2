"""
Figure 7: RISE Projection Boxplots by Cell Type

This figure runs RISE (pf2_nd), stacks the projections, and for a chosen cell type,
plots boxplots of the projection values for each rank.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from parafac2.parafac2 import pf2_nd
from ..import_data import import_balf_covid, add_cond_idxs


def makeFigure():
    # Parameters
    rank = 30
    celltype_of_interest = "Macrophages"  # Change as needed

    # Load and prepare data
    print("Importing BALF COVID-19 data...")
    adata = import_balf_covid()
    adata = add_cond_idxs(adata, "sample")
    print("Loaded data with shape:", adata.shape)

    # Run RISE (pf2_nd)
    print("Running RISE (pf2_nd)...")
    projections, _ = pf2_nd(adata, rank=rank, n_iter_max=100, tol=1e-6, random_state=42)
    # projections is a list of arrays, one per condition, each shape (cells_in_condition, rank)

    # Stack projections: shape (rank, total_cells)
    stacked = np.concatenate(
        [proj.T for proj in projections], axis=1
    )  # (rank, total_cells)

    # Get cell type labels for all cells (in the same order as stacking)
    celltype_labels = np.concatenate(
        [
            adata.obs.loc[adata.obs["condition_unique_idxs"] == i, "celltype"].values
            for i in range(len(projections))
        ]
    )

    # For each rank, get the projection values for the chosen cell type
    fig, axes = plt.subplots(
        1, rank, figsize=(2 * rank, 6), sharey=True, constrained_layout=True
    )
    for r in range(rank):
        values = stacked[r, :]
        mask = celltype_labels == celltype_of_interest
        axes[r].boxplot(values[mask])
        axes[r].set_title(f"Rank {r+1}")
        axes[r].set_xticks([])
        if r == 0:
            axes[r].set_ylabel(f"{celltype_of_interest} projection value")
    fig.suptitle(f"RISE Projections for '{celltype_of_interest}' by Rank", fontsize=16)
    return fig
