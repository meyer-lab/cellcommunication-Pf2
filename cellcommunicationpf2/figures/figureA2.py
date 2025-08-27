"""
Figure A2: Non-negative CP for COVID-19 pseudobulk
"""

import numpy as np
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
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors
)

from .commonFuncs.plotGeneral import (
    rotate_yaxis
)


def makeFigure():
    ax, f = getSetup((24, 12), (2, 4))
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

    groupby = "celltype"
    groupby_names = X_filtered.obs[groupby].unique()
    types = ["mean", "fraction"]
    for i, t in enumerate(types):
        appended_pseudobulk = pseudobulk_X(X_filtered, condition_name=condition_column, groupby=groupby, type=t)
        interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs)

        _, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=10000, tol=1e-9)
        # Confirm cpd factors are only positive
        for factor in cpd_factors:
          assert np.all(factor >= 0), "CPD factors contain negative values"
            
        X_filtered.uns["Pf2_A"] = cpd_factors[0]  # Condition factor
        X_filtered.uns["Pf2_B"] = cpd_factors[1]  # Sender cell types factor
        X_filtered.uns["Pf2_C"] = cpd_factors[2]  # Receiver cell types factor
        X_filtered.uns["Pf2_D"] = cpd_factors[3]  # LR pairs factor
        X_filtered.uns["Pf2_lr_pairs"] = filtered_lr_pairs  # LR pairs

        plot_condition_factors(
            data=X_filtered,
            ax=ax[(4*i)],
            cond=condition_column,
            cond_group_labels=sample_to_group,
            group_cond=True,
        )
  
        plot_eigenstate_factors(
            data=X_filtered,
            ax=ax[(4*i+1)],
            factor_type="Pf2_B",
        )
        ax[(4*i+1)].set_yticklabels(groupby_names)

        plot_eigenstate_factors(
            data=X_filtered,
            ax=ax[(4*i+2)],
            factor_type="Pf2_C",
        )
        ax[(4*i+2)].set_yticklabels(groupby_names)

        plot_lr_factors(
            data=X_filtered,
            ax=ax[(4*i+3)],
            weight=0.06,
        )

        ax[(4*i)].set_title("Conditions Factor")
        ax[(4*i+1)].set_title("Sender Cell Type Factor")
        ax[(4*i+2)].set_title("Receiver Cell Type Factor")
        ax[(4*i+3)].set_title("Ligand-Receptor Factor")
        rotate_yaxis(ax[(4*i+1)], rotation=0)
        rotate_yaxis(ax[(4*i+2)], rotation=0)


    return f