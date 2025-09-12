"""
Figure A4c: Decomposition of the communication data and tensor for BAL data.
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
    pseudobulk_X, 
    load_tensor
)
from ..cc_pf2 import (
    calc_communication_score_pseudobulk,
    pseudobulk_nncp_decomposition,
    save_ccc_rise_results
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
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    tc2c_tensor = load_tensor("cellcommunicationpf2/data/Tensor-cell2cell/tensor-bal.pkl")
    X = import_balf_covid(gene_threshold=0, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()
    groupby = "celltype"
    condition_column = "sample"
    group_col = "condition"
    X = add_cond_idxs(X, condition_column)
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
 
    X = X[X.obs[groupby].isin(tc2c_tensor.order_names[2])]
    # type = "fraction"
    type = "mean"

    appended_pseudobulk = pseudobulk_X(X, condition_name=condition_column, groupby=groupby, type=type)
    valid_tc2c_pairs = set(tc2c_tensor.order_names[1])
    lr_pairs_filtered = lr_pairs[lr_pairs["interaction_symbol"].isin(valid_tc2c_pairs)].reset_index(drop=True)

    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs, complex_sep="&")
    
    cpd_weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=100000, tol=1e-11, random_state=0)

    save_ccc_rise_results(X, cpd_factors, cpd_weights, filtered_lr_pairs)

    plot_condition_factors(
        data=X,
        ax=ax[0],
        cond="sample",
        cond_group_labels=sample_to_group,
        group_cond=True
    )
    plot_eigenstate_factors(    
        data=X,
        ax=ax[1],
        factor_type="Pf2_B"
    )

    celltype_names = np.unique(X.obs[groupby])
    ax[1].set_yticklabels(celltype_names)
    rotate_yaxis(ax[1], 0)

    plot_eigenstate_factors(
        data=X,
        ax=ax[2],
        factor_type="Pf2_C",
    )
    ax[2].set_yticklabels(celltype_names)
    rotate_yaxis(ax[2], 0)

    plot_lr_factors(
        data=X,
        ax=ax[3],
        weight=0.16
    )

    
    return f