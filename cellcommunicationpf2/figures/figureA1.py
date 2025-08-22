"""
Figure 1: CC-PF2 Factor Visualization

XXXX
"""

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import (
    subplotLabel,
    getSetup,
)
from ..utils import (
    pseudobulk_X
)
from ..cc_pf2 import (
    calc_communication_score_pseudobulk,
    pseudobulk_nncp_decomposition,
    standardize_cp_decomposition
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
    rotate_yaxis
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((18, 18), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing data...")
    X = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    # Create a mapping from each sample to its corresponding condition (e.g., 'severe')
    # This will be used for grouping and coloring the heatmap
    group_col = "condition"
    sample_to_group = X_filtered.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

    print(X_filtered)
    groupby = "celltype"
    appended_pseudobulk  = pseudobulk_X(X_filtered, condition_name=condition_column, groupby=groupby)
    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk)

    print(interaction_tensor.shape)
    print(filtered_lr_pairs)

    cpd_weights, cpd_factors, r2x = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=10000, tol=1e-9)
    # cpd_weights, cpd_factors = standardize_cp_decomposition(cpd_weights, cpd_factors)

    plot_condition_factors(
        data=cpd_factors[0],
        ax=ax[0],
        condition_labels=X.obs[condition_column].unique(),
        cond=group_col,
        cond_group_labels=sample_to_group,
        group_cond=True,
        vmin=0
    )
  

    plot_eigenstate_factors(
        data=cpd_factors[1],
        ax=ax[1],
        factor_type="Sender Cell Type",
        labels=X.obs[groupby].unique(),
        vmin=0
    )
    
    plot_eigenstate_factors(
        data=cpd_factors[2],
        ax=ax[2],
        factor_type="Receiver Cell Type",
        labels=X.obs[groupby].unique(),
        vmin=0 
    )

    plot_lr_factors(
        data=cpd_factors[3],
        ax=ax[3],
        lr_pairs=filtered_lr_pairs,
        weight=0.04,
        vmin=0
    )
    
    ax[0].set_title("Conditions Factor")
    ax[1].set_title("Sender Cell Type Factor")
    ax[2].set_title("Receiver Cell Type Factor")
    ax[3].set_title("Ligand-Receptor Factor")
    rotate_yaxis(ax[1], rotation=0)
    rotate_yaxis(ax[2], rotation=0)

    return f





