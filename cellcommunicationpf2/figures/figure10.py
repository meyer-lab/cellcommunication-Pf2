"""
Figure 10: Weighted Projections from CC-PF2 Sender and Receiver Eigenstates

This figure shows PaCMAP projections weighted by sender and receiver eigenstate 
components from the CC-PF2 decomposition to identify communication patterns.
"""

import numpy as np
from matplotlib import pyplot as plt
import pacmap

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..utils import run_cc_pf2_workflow
from .commonFuncs.plotPaCMAP import plot_labels_pacmap, plot_wp_pacmap


def makeFigure():
    """Generate Figure 10 showing PaCMAP weighted by CC-PF2 sender and receiver eigenstates."""

    # Import and prepare data
    print("Importing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata, condition_column)

    # Parameters for CC-PF2
    rise_rank = 30
    cp_rank = 10
    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    # Run CC-PF2 workflow
    print(f"Running CC-PF2 with rank={rise_rank} and cp_rank={cp_rank}...")
    adata_filtered, r2x = run_cc_pf2_workflow(
        adata_filtered,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )

    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")

    # Get the raw PARAFAC2 projections and CC-PF2 eigenstate factors
    stacked_projections = np.vstack(adata_filtered.uns["Pf2_projections"])
    sender_factor = adata_filtered.uns["Pf2_B"]    # Sender eigenstate matrix (rank x cp_rank)
    receiver_factor = adata_filtered.uns["Pf2_C"]  # Receiver eigenstate matrix (rank x cp_rank)

    # Calculate sender and receiver weighted projections
    sender_weighted_projections = stacked_projections @ sender_factor
    receiver_weighted_projections = stacked_projections @ receiver_factor

    # Store weighted projections
    adata_filtered.obsm["Pf2_sender_weighted_projections"] = sender_weighted_projections
    adata_filtered.obsm["Pf2_receiver_weighted_projections"] = receiver_weighted_projections

    # Compute PaCMAP embeddings using the RAW stacked projections
    print("Computing PaCMAP embeddings...")
    embedding = pacmap.PaCMAP(n_neighbors=13, random_state=random_state)
    pacmap_projections = embedding.fit_transform(stacked_projections)  # Use raw projections for embedding
    adata_filtered.obsm["Pf2_PaCMAP_projections"] = pacmap_projections
    print("PaCMAP computation complete.")

    # Create figure: cell types + sender components + receiver components
    n_components = cp_rank  # 10 components

    # Use gridspec for custom sizing - 3 sections: cell type, sender, receiver
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(25, 12), constrained_layout=True)
    gs = GridSpec(2, 11, figure=fig, 
                  height_ratios=[1, 1], 
                  width_ratios=[2] + [1]*10)  # First column wider for cell type

    # Large cell type plot (first column, spans all rows)
    ax_celltype = fig.add_subplot(gs[:, 0])
    plot_labels_pacmap(
        adata_filtered,
        labelType="celltype",
        ax=ax_celltype,
        cmap="tab20"
    )
    ax_celltype.set_title("Cell Types", fontsize=14, pad=20)

    # First row: Sender eigenstate components
    for comp in range(1, n_components + 1):
        ax_sender = fig.add_subplot(gs[0, comp])
        
        # Temporarily set the sender weighted projections for plotting
        adata_filtered.obsm["Pf2_cell_cell_weighted_projections"] = sender_weighted_projections
        
        plot_wp_pacmap(
            adata_filtered,
            cmp=comp,
            ax=ax_sender,
            cbarMax=np.max(np.abs(sender_weighted_projections[:, comp - 1]))
        )
        ax_sender.set_title(f"Sender {comp}", fontsize=10)
        ax_sender.tick_params(axis='both', which='major', labelsize=8)

    # Second row: Receiver eigenstate components  
    for comp in range(1, n_components + 1):
        ax_receiver = fig.add_subplot(gs[1, comp])
        
        # Temporarily set the receiver weighted projections for plotting
        adata_filtered.obsm["Pf2_cell_cell_weighted_projections"] = receiver_weighted_projections
        
        plot_wp_pacmap(
            adata_filtered,
            cmp=comp,
            ax=ax_receiver,
            cbarMax=np.max(np.abs(receiver_weighted_projections[:, comp - 1]))
        )
        ax_receiver.set_title(f"Receiver {comp}", fontsize=10)
        ax_receiver.tick_params(axis='both', which='major', labelsize=8)

    # Add row labels
    fig.text(0.02, 0.75, 'Sender\nEigenstates', rotation=90, va='center', ha='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.25, 'Receiver\nEigenstates', rotation=90, va='center', ha='center', fontsize=12, weight='bold')

    # Add overall figure title
    plt.suptitle(
        f"CC-PF2 Weighted Projections: Sender vs Receiver Eigenstates (R²X = {r2x:.4f})",
        fontsize=16, y=0.98
    )

    print("Figure generation complete.")
    return fig