"""
Figure 8: PaCMAP Visualization for Cell Type and Component Analysis

This figure shows PaCMAP projections colored by cell type and weighted by 
component projections to identify which components correspond to specific cell types.
"""

import numpy as np
from matplotlib import pyplot as plt
import pacmap
from parafac2.parafac2 import parafac2_nd

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
)
from .common import subplotLabel
from .commonFuncs.plotPaCMAP import plot_labels_pacmap, plot_wp_pacmap


def makeFigure():
    """Generate Figure 8 showing PaCMAP visualizations for cell type and component analysis."""

    # Import and prepare data
    print("Importing data...")
    adata = import_balf_covid()

    # Add numerical indices for each patient sample
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata, condition_column)

    # Parameters for PARAFAC2
    rank = 30
    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    # Run PARAFAC2 decomposition
    print(f"Running PARAFAC2 with rank={rank}...")
    pf2_output, r2x = parafac2_nd(
        adata_filtered,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )

    print(f"PARAFAC2 decomposition R2X: {r2x:.4f}")

    # Stack cell projections from all conditions
    _, factors, projections = pf2_output
    # Projections are a list of (cells_in_condition, rank) matrices
    # We stack them to get (total_cells, rank)
    stacked_projections = np.vstack(projections)

    eigenstate_factor = factors[1]  # This should be (rank x rank)
    weighted_projections = stacked_projections @ eigenstate_factor

    # Store both the raw projections and weighted projections
    adata_filtered.obsm["Pf2_cell_projections"] = stacked_projections
    adata_filtered.obsm["Pf2_cell_cell_weighted_projections"] = stacked_projections

    # Compute PaCMAP with n_neighbors=13
    print("Computing PaCMAP with n_neighbors=13...")
    embedding = pacmap.PaCMAP(n_neighbors=13, random_state=random_state)
    pacmap_projections = embedding.fit_transform(stacked_projections)
    adata_filtered.obsm["Pf2_PaCMAP_projections"] = pacmap_projections
    print("PaCMAP computation complete.")

    # Create figure with larger cell type plot and smaller component plots
    n_components = rank  # 30 components

    # Use gridspec for custom sizing
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs = GridSpec(6, 6, figure=fig, height_ratios=[2, 1, 1, 1, 1, 1], width_ratios=[2, 1, 1, 1, 1, 1])

    # Large cell type plot (top left, 2x2)
    ax_celltype = fig.add_subplot(gs[0:2, 0:2])
    plot_labels_pacmap(
        adata_filtered,
        labelType="celltype",
        ax=ax_celltype,
        cmap="tab20"
    )

    # Smaller component plots
    ax_components = []
    positions = [(i, j) for i in range(6) for j in range(6) if not (i < 2 and j < 2)]

    for comp in range(1, min(n_components + 1, len(positions) + 1)):
        row, col = positions[comp - 1]
        ax_comp = fig.add_subplot(gs[row, col])
        ax_components.append(ax_comp)

        plot_wp_pacmap(
            adata_filtered,
            cmp=comp,
            ax=ax_comp,
            cbarMax=0.2
        )
        ax_comp.set_title(f"Eigenstate {comp}", fontsize=10)

    print("Figure generation complete.")
    return fig
