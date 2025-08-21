"""
Figure 1: CC-PF2 Factor Visualization

This figure shows the factors from a rank-10 CC-PF2 decomposition
of the BALF COVID-19 dataset.
"""

import numpy as np 
import pandas as pd
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
    run_cc_pf2_workflow,
    pseudobulk_X
)
from ..cc_pf2 import (
    calc_communication_score_pseudobulk,
    pseudobulk_cp_decomposition,
    standardize_cp_decomposition
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
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

    cpd_weights, cpd_factors, r2x = pseudobulk_cp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=100, tol=1e-6, random_state=42)
    cpd_weights, cpd_factors = standardize_cp_decomposition(cpd_weights, cpd_factors)

    plot_condition_factors(
        data=cpd_factors[0],
        ax=ax[0],
        condition_labels=X.obs[condition_column].unique(),
        cond=group_col,
        cond_group_labels=sample_to_group,
        group_cond=True
    )
    # ax[0].set_title("Factor 0: Patient Conditions")

    return f





