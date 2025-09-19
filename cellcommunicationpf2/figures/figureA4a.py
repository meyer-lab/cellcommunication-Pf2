"""
Figure A4a: Decomposition of the pseudobulk communication data and tensor for BAL data.
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
from ..ccc_rise import (
    calc_communication_score_pseudobulk,
    pseudobulk_cp_decomposition,
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
import anndata

def makeFigure():
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    # X = import_balf_covid(gene_threshold=0.001, normalize=True)
    # lr_pairs = import_ligand_receptor_pairs()
    # groupby = "celltype"
    # condition_column = "sample"
    # group_col = "condition"
    # X = add_cond_idxs(X, condition_column)
    # type = "mean"

    # appended_pseudobulk = pseudobulk_X(X, condition_name=condition_column, groupby=groupby, type=type)
    # interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs, complex_sep="&")

    # print(np.shape(interaction_tensor))
    # cp_rank = 8
    # cpd_weights, cpd_factors, _ = pseudobulk_cp_decomposition(interaction_tensor, cp_rank=cp_rank, n_iter_max=1000, tol=1e-11)

    # save_ccc_rise_results(X, cpd_factors, cpd_weights, filtered_lr_pairs)
    
    # X = X.write_h5ad("cellcommunicationpf2/data/bal/bal_pseudobulk.h5ad")
    
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19_pseudobulk.h5ad")
    groupby = "celltype"
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

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
        factor_type="B"
    )

    celltype_names = np.unique(X.obs[groupby])
    ax[1].set_yticklabels(celltype_names)
    rotate_yaxis(ax[1], 0)

    plot_eigenstate_factors(
        data=X,
        ax=ax[2],
        factor_type="C",
    )
    ax[2].set_yticklabels(celltype_names)
    rotate_yaxis(ax[2], 0)

    plot_lr_factors(
        data=X,
        ax=ax[3],
        weight=0.03
    )
    

    return f