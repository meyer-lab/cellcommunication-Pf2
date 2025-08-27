"""
Figure A2: XXXX
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
# from .commonFuncs.plotFactors import (
#     plot_condition_factors,
#     plot_eigenstate_factors,
#     plot_lr_factors,
#     rotate_yaxis
# )


def makeFigure():
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

    groupby = "celltype"
    appended_pseudobulk  = pseudobulk_X(X_filtered, condition_name=condition_column, groupby=groupby, type="mean")
    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs)


    print(interaction_tensor.shape)
    print(filtered_lr_pairs)

    _, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=10000, tol=1e-9)
    # Confirm cpd factors are only positive
    for factor in cpd_factors:
        assert np.all(factor >= 0), "CPD factors contain negative values"

    return f




