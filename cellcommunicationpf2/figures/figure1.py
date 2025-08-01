"""
Figure 1: CC-PF2 Factor Visualization

This figure shows the factors from a rank-10 CC-PF2 decomposition
of the BALF COVID-19 dataset.
"""

from matplotlib import pyplot as plt

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import subplotLabel
from ..utils import run_cc_pf2_workflow
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

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata, condition_column)

    # Create a mapping from each sample to its corresponding condition (e.g., 'severe')
    # This will be used for grouping and coloring the heatmap
    group_col = "condition"
    sample_to_group = adata_filtered.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

    # Parameters for CC-PF2
    rise_rank = 30
    cp_rank = 10
    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    # Run CC-PF2
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
        f"CC-PF2 Decomposition (Rank {rise_rank}, R²X = {r2x:.4f})", fontsize=16, y=0.98
    )
    print("Figure generation complete.")
    return fig
