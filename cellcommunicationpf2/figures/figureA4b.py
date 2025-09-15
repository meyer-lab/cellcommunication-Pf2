"""
Figure A4b: Decomposition of the communication tensor from Tensorcell2cell.
"""

import numpy as np
from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import (
    subplotLabel,
    getSetup,
)
from ..utils import (
    load_tensor
)
from ..cc_pf2 import (
    save_ccc_rise_results,
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
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    cond_names = np.unique(["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"])

    tc2c_tensor = load_tensor("cellcommunicationpf2/data/Tensor-cell2cell/tensor-bal.pkl")
    lr_pairs = import_ligand_receptor_pairs()
    
    tc2c_tensor_only, lr_pairs_filtered = filter_tensor(tc2c_tensor, lr_pairs, cond_names)

    cpd_weights, cpd_factors, _ = pseudobulk_nncp_decomposition(tc2c_tensor_only, cp_rank=10, n_iter_max=1000, tol=1e-11, random_state=0)

    cpd_factors = [cpd_factors[0], cpd_factors[2], cpd_factors[3], cpd_factors[1]]
    X = import_balf_covid(gene_threshold=0, normalize=False)
    X = save_ccc_rise_results(X, cpd_factors, cpd_weights, lr_pairs_filtered)


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
        factor_type="Pf2_B",
    )
    tc2c_celltype_names = np.unique(tc2c_tensor.order_names[2])
    ax[1].set_yticklabels(tc2c_celltype_names)
    rotate_yaxis(ax[1], 0)

    plot_eigenstate_factors(
        data=X,
        ax=ax[2],
        factor_type="Pf2_C",
    )
    ax[2].set_yticklabels(tc2c_celltype_names)
    rotate_yaxis(ax[2], 0)

    plot_lr_factors(
        data=X,
        ax=ax[3],
        weight=0.16
    )

    return f





def filter_tensor(tc2c_tensor, lr_pairs, cond_names):
    """Change order of tensor to reorder with conditions, cell types, and filter LR pairs"""
    # Change conditions order to be cond_names
    tensor = tc2c_tensor.tensor
    tc2c_condition_names = ["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"]
    order_indices = [tc2c_condition_names.index(name) for name in cond_names]
    tensor = np.take(tensor, order_indices, axis=0)
    
    # Change cell type order
    tc2c_celltype_names = tc2c_tensor.order_names[2]
    celltype_order = np.unique(tc2c_celltype_names)
    order_indices = [list(tc2c_celltype_names).index(name) for name in celltype_order]
    tensor = tensor[:, :, order_indices, :]
    tensor = tensor[:, :, :, order_indices]
    
    # Change LR pair order and filter to only valid LR pairs
    valid_tensor_pairs = set(lr_pairs['interaction_symbol'])
    tensor_lrpair_names = np.array(tc2c_tensor.order_names[1])
    keep_indices = [i for i, pair in enumerate(tensor_lrpair_names) if pair in valid_tensor_pairs]
    tensor = np.take(tensor, keep_indices, axis=1)
    tensor_lrpair_names = tensor_lrpair_names[keep_indices]
    lrpair_to_index = {name: idx for idx, name in enumerate(tensor_lrpair_names)}
    interaction_tensor_lrpair = lr_pairs["interaction_symbol"].values
    order_indices = [lrpair_to_index[name] for name in interaction_tensor_lrpair if name in lrpair_to_index]
    tensor = tensor[:, order_indices, :, :]
    new_lr_pairs = lr_pairs[lr_pairs["interaction_symbol"].isin(tensor_lrpair_names)].reset_index(drop=True)
    new_lr_pairs = new_lr_pairs.reindex(order_indices)
    
    assert new_lr_pairs["interaction_symbol"].tolist() == tensor_lrpair_names[order_indices].tolist(), "LR pairs order does not match tensor"

    return tensor, new_lr_pairs