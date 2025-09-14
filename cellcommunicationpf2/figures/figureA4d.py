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
    plot_lr_factors_partial
)

from .commonFuncs.plotGeneral import (
    rotate_yaxis
)

def makeFigure():
    ax, f = getSetup((12, 12), (5, 4))
    subplotLabel(ax)

    tc2c_tensor = load_tensor("cellcommunicationpf2/data/Tensor-cell2cell/tensor-bal.pkl")
    X = import_balf_covid(gene_threshold=0, normalize=True)
    print(X)
    lr_pairs = import_ligand_receptor_pairs()
    groupby = "celltype"
    condition_column = "sample"
    group_col = "condition"
    X = add_cond_idxs(X, condition_column)
    X = X[X.obs[groupby].isin(tc2c_tensor.order_names[2])]
    # type = "fraction"
    type = "mean"

    appended_pseudobulk = pseudobulk_X(X, condition_name=condition_column, groupby=groupby, type=type)
    # valid_tc2c_pairs = set(tc2c_tensor.order_names[1])
    # lr_pairs = lr_pairs[lr_pairs["interaction_symbol"].isin(valid_tc2c_pairs)].reset_index(drop=True)
    
    print(lr_pairs)

    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs, complex_sep="&")
    print(filtered_lr_pairs)

    print(np.shape(interaction_tensor))
    cp_rank = 10
    cpd_weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=cp_rank, random_state=0, n_iter_max=1000)

    X = save_ccc_rise_results(X, cpd_factors, cpd_weights, filtered_lr_pairs)
    
      
    for i in range(cp_rank):
        plot_lr_factors_partial(X, i, ax[2*i], geneAmount=10, top=True)
        plot_lr_factors_partial(X, i, ax[2*i+1], geneAmount=10, top=False)

    
    return f