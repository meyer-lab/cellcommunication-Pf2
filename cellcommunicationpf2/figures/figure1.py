"""
Figure 1: CC-PF2 Factor Visualization

This figure shows the factors from a rank-10 CC-PF2 decomposition
of the BALF COVID-19 dataset.
"""

from matplotlib import pyplot as plt

from ..cc_pf2 import cc_pf2, standardize_cc_pf2
from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import subplotLabel
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)


def makeFigure():
    """Generate Figure 1 showing CC-PF2 factor heatmaps using specialized plotting functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    ax = axes.flatten()
    subplotLabel(ax)

    # Import and prepare data
    print("Importing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    print("Filtering data...")
    adata_filtered, lr_pairs_filtered = anndata_lrp_overlap(adata, lr_pairs)

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata_filtered, condition_column)

    # Create a mapping from each sample to its corresponding condition (e.g., 'severe')
    # This will be used for grouping and coloring the heatmap
    group_col = "condition"
    sample_to_group = adata_filtered.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

    # Parameters for CC-PF2
    rank = 10
    n_iter_max = 100
    tol = 1e-3
    random_state = 42

    # Run CC-PF2
    print(f"Running CC-PF2 with rank={rank}...")
    results, r2x = cc_pf2(
        adata_filtered, rank, n_iter_max, tol, random_state=random_state
    )
    cp_results, projections = results
    cp_weights, factors = cp_results

    print("Standardizing factors...")
    _, factors, projections = standardize_cc_pf2(
        factors, projections, weights=cp_weights
    )

    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")

    # Store factors in AnnData object for easy access by plotting functions
    adata_filtered.uns["Pf2_A"] = factors[0]  # Condition factor
    adata_filtered.uns["Pf2_B"] = factors[1]  # Sender cells factor
    adata_filtered.uns["Pf2_C"] = factors[2]  # Receiver cells factor
    adata_filtered.uns["Pf2_D"] = factors[3]  # LR pairs factor
    adata_filtered.uns["Pf2_lr_pairs"] = lr_pairs_filtered.reset_index(drop=True)

    # Generate heatmaps for each factor
    print("Generating heatmaps...")

    # Factor 0: Patient Conditions (Samples)
    plot_condition_factors(
        adata_filtered,
        ax[0],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,  # Sort samples by their condition group
    )
    ax[0].set_title("Factor 0: Patient Conditions")

    # Factor 1: Sender Cell Eigenstates
    plot_eigenstate_factors(adata_filtered, ax[1], factor_type="Pf2_B")
    ax[1].set_title("Factor 1: Sender Cell Eigenstates")

    # Factor 2: Receiver Cell Eigenstates
    plot_eigenstate_factors(adata_filtered, ax[2], factor_type="Pf2_C")
    ax[2].set_title("Factor 2: Receiver Cell Eigenstates")

    # Factor 3: Ligand-Receptor Pairs
    plot_lr_factors(adata_filtered, ax[3], trim=True)
    ax[3].set_title("Factor 3: LR Pairs")

    # Add overall figure title with R2X information
    plt.suptitle(
        f"CC-PF2 Decomposition (Rank {rank}, R²X = {r2x:.4f})", fontsize=16, y=0.98
    )
    print("Figure generation complete.")
    return fig
