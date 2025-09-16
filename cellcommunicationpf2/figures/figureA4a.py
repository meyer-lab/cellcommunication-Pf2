"""
Figure A4a: Decomposition of the communication data from Tensorcell2cell.
"""

import numpy as np
import pandas as pd
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
from ..ccc_rise import (
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

    cond_names = np.unique(["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"])
    celltypes = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    total_df = []
    for _, cond_name in enumerate(cond_names):
        df = pd.read_csv(f"cellcommunicationpf2/data/Tensor-cell2cell/{cond_name}.csv")
        df.set_index(df.columns[0], inplace=True)
        df = df[celltypes]
        total_df.append(df.fillna(0))

    tc2c_tensor = load_tensor("cellcommunicationpf2/data/Tensor-cell2cell/tensor-bal.pkl")
    lr_pairs = import_ligand_receptor_pairs()

    valid_tc2c_pairs = set(tc2c_tensor.order_names[1])
    lr_pairs_filtered = lr_pairs[lr_pairs["interaction_symbol"].isin(valid_tc2c_pairs)].reset_index(drop=True)
    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(total_df, lr_pairs=lr_pairs_filtered, complex_sep="&")

    cpd_weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=1000, tol=1e-11, random_state=0)

    X = import_balf_covid(gene_threshold=0, normalize=False)

    X = save_ccc_rise_results(X, cpd_factors, cpd_weights, filtered_lr_pairs)

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




