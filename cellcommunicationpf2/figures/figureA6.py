"""
Figure A6: Comparison of the factors obtained from the pseudobulk decomposition and the tensor decomposition for the BALF COVID dataset.
"""

from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
import numpy as np
import pandas as pd
import pickle
import os
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
    
    tc2c_tensor = load_tensor("Tensor-BALF.pkl")

    print("Importing data...")
    X = import_balf_covid(gene_threshold=0)
    
    lr_pairs = import_ligand_receptor_pairs()
        
    # Override ligand and receptor columns using interaction_name_2
    if 'interaction_name_2' in lr_pairs.columns:
        lr_pairs['ligand'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[0].upper())
        lr_pairs['receptor'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[1].upper().replace('(', '').replace(')', '').replace('+', '&'))
    # Also update interaction_symbol if present
    if 'interaction_symbol' in lr_pairs.columns:
        lr_pairs['interaction_symbol'] = lr_pairs['interaction_symbol'].str.upper().str.replace('_', '&')
    
    only_in_t2c2_tensor = {
    'ITGB2^CD226', 'TGFB1^ACVR1&TGFBR1', 'LTA^LTB&LTBR', 'ITGB2^ICAM2',
    'SEMA3C^NRP1&NRP2', 'ITGB2^ICAM1', 'EBI3^IL27RA&IL6ST', 'TNFSF10^TNFRSF10B'
}
    present = []
    not_present = []
    for pair in only_in_t2c2_tensor:
        if pair in lr_pairs["interaction_symbol"].values:
            present.append(pair)
        else:
            not_present.append(pair)

    print("Present in lr_pairs:", present)
    print("Not present in lr_pairs:", not_present)
    
    cell_types = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    
    X = X[X.obs["celltype"].isin(cell_types)]
    print(np.unique(X.obs["celltype"].unique()))

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    groupby = "celltype"
    groupby_names = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    
    types = ["fraction"]
    # types = ["mean", "fraction"]


    for i, t in enumerate(types):
        appended_pseudobulk = pseudobulk_X(X_filtered, condition_name=condition_column, groupby=groupby, type=t)
        valid_tc2c_pairs = set(tc2c_tensor.order_names[1])
        lr_pairs_filtered = lr_pairs[lr_pairs['interaction_symbol'].isin(valid_tc2c_pairs)].reset_index(drop=True)
        

        interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs_filtered, complex_sep="&")

        print(interaction_tensor.shape)
        overlap = set(filtered_lr_pairs["interaction_symbol"].values) & set(tc2c_tensor.order_names[1])
        print("Overlapping LR pairs:", len(overlap))
        print("Length of tensor order names:", len(tc2c_tensor.order_names[1]))
        print("Length of filtered LR pairs:", filtered_lr_pairs.shape[0])

        only_in_tensor = set(tc2c_tensor.order_names[1]) - set(filtered_lr_pairs["interaction_symbol"].values)
        print("Only in tensor_order_names:", only_in_tensor)
        
        
        valid_tensor_pairs = set(filtered_lr_pairs['interaction_symbol'])
        tensor_lr_pairs = np.array(tc2c_tensor.order_names[1])
        keep_indices = [i for i, pair in enumerate(tensor_lr_pairs) if pair in valid_tensor_pairs]
        # Filter the tensor along the LR pair mode (last mode)
        tc2c_tensor_filtered = np.take(np.array(tc2c_tensor.tensor), keep_indices, axis=1)

        
        # Rearrange interaction tensor to be in the same order the tensor loaded from file, so we can directly use it for decomposition. Information are arrays
        tc2c_condition_names = ["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"]
        interaction_tensor_cond = ["C51", "C52", "C100", "C141", "C142", "C143", "C144", "C145", "C146","C148", "C149", "C152"]

        order_indices = [interaction_tensor_cond.index(name) for name in tc2c_condition_names]
        print(order_indices)

        interaction_tensor = interaction_tensor[order_indices, :, :, :]
        
        # Rearrange interaction tensor to be in the same order the tensor loaded from file, so we can directly use it for decomposition. 
        tc2c_celltype_names = tc2c_tensor.order_names[2]
        interaction_tensor_celltype = groupby_names
        order_indices = [tc2c_celltype_names.index(name) for name in interaction_tensor_celltype]
        print("Interaction tensor celltype names:", interaction_tensor_celltype)
        interaction_tensor = interaction_tensor[:, order_indices, :, :]
        print(order_indices)
        print(np.shape(interaction_tensor))
        
        interaction_tensor = interaction_tensor[:, :, order_indices, :]
        print(np.shape(interaction_tensor))
        
        # Rearrange interaction tensor to be in the same order the tensor loaded from file, so we can directly use it for decomposition. 
        tc2c_lrpair_names = tc2c_tensor.order_names[1]
        tc2c_lrpair_names = np.array(tc2c_lrpair_names)[keep_indices]
        # print("Interaction tensor LR pair names:", tc2c_lrpair_names)
        # print(np.shape(keep_indices))
        
        print(tc2c_lrpair_names)
        interaction_tensor_lrpair = filtered_lr_pairs["interaction_symbol"].values
        # print(interaction_tensor_lrpair)
        lrpair_to_index = {name: idx for idx, name in enumerate(tc2c_lrpair_names)}
        order_indices = [lrpair_to_index[name] for name in interaction_tensor_lrpair]
        # print(order_indices)
        # print(np.shape(order_indices))
        
        
    
        interaction_tensor = interaction_tensor[:, :, :, order_indices]

        weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=15, n_iter_max=100000, tol=1e-11, random_state=0)
        # Confirm cpd factors are only positive
        for factor in cpd_factors:
            assert np.all(factor >= 0), "CPD factors contain negative values"

        X_filtered.uns["Pf2_A"] = cpd_factors[0]  # Condition factor
        X_filtered.uns["Pf2_B"] = cpd_factors[1]  # Sender cell types factor
        X_filtered.uns["Pf2_C"] = cpd_factors[2]  # Receiver cell types factor
        X_filtered.uns["Pf2_D"] = cpd_factors[3]  # LR pairs factor
        X_filtered.uns["Pf2_lr_pairs"] = filtered_lr_pairs  # LR pairs

        X_filtered.uns["Pf2_weights"] = weights


        plot_condition_factors(
            data=X_filtered,
            ax=ax[(4*i)],
            cond=condition_column,
            yt = tc2c_condition_names
            # cond_group_labels=sample_to_group,
            # group_cond=True,
        )
  
        plot_eigenstate_factors(
            data=X_filtered,
            ax=ax[(4*i+1)],
            factor_type="Pf2_B",
        )
        ax[(4*i+1)].set_yticklabels(tc2c_celltype_names)

        plot_eigenstate_factors(
            data=X_filtered,
            ax=ax[(4*i+2)],
            factor_type="Pf2_C",
        )
        ax[(4*i+2)].set_yticklabels(tc2c_celltype_names)

        plot_lr_factors(
            data=X_filtered,
            ax=ax[(4*i+3)],
            weight=0.18,
        )

        ax[(4*i)].set_title("Conditions Factor")
        ax[(4*i+1)].set_title("Sender Cell Type Factor")
        ax[(4*i+2)].set_title("Receiver Cell Type Factor")
        ax[(4*i+3)].set_title("Ligand-Receptor Factor")
        rotate_yaxis(ax[(4*i+1)], rotation=0)
        rotate_yaxis(ax[(4*i+2)], rotation=0)

    # Save X_filtered for the tensor decomposition
    X_filtered_tensor = X_filtered.copy()
    
    cp_rank = 10  # You can adjust this rank as needed
    n_iter_max = 10000
    tol = 1e-9
    
    print("Filtered tensor shape:", tc2c_tensor_filtered.shape)
    print("Performing non-negative CP decomposition on the loaded tensor...")
    nncp_weights, nncp_factors, r2x = pseudobulk_nncp_decomposition(
        tc2c_tensor_filtered,
        cp_rank=15, n_iter_max=100000, tol=1e-11, random_state=0
    )
    for factor in nncp_factors:
        assert np.all(factor >= 0), "CPD factors contain negative values"

    X_filtered_tensor.uns["Pf2_A"] = nncp_factors[0]  # Condition factor
    X_filtered_tensor.uns["Pf2_B"] = nncp_factors[2]  # Sender cell types factor
    X_filtered_tensor.uns["Pf2_C"] = nncp_factors[3]  # Receiver cell types factor
    X_filtered_tensor.uns["Pf2_D"] = nncp_factors[1]  # LR pairs factor
    X_filtered_tensor.uns["Pf2_lr_pairs"] = filtered_lr_pairs  # LR pairs
    X_filtered_tensor.uns["Pf2_weights"] = nncp_weights
         
    print(tc2c_tensor.order_names[0])
    print(tc2c_tensor.order_names[2])
    print(tc2c_tensor.order_names[3])
    
    print(condition_column)
    # print(sample_to_group)
    plot_condition_factors(
        data=X_filtered_tensor,
        ax=ax[4],
        cond=condition_column,
        yt=tc2c_condition_names
        # cond_group_labels=sample_to_group,
        # group_cond=True,
    )
    plot_eigenstate_factors(
        data=X_filtered_tensor,
        ax=ax[5],
        factor_type="Pf2_B",
    )
    ax[5].set_yticklabels(tc2c_celltype_names)
    plot_eigenstate_factors(
        data=X_filtered_tensor,
        ax=ax[6],
        factor_type="Pf2_C",
    )
    ax[6].set_yticklabels(tc2c_celltype_names)
    plot_lr_factors(
        data=X_filtered_tensor,
        ax=ax[7],
        weight=0.18,
    )
    
    ax[4].set_title("Tensor: Conditions Factor")
    ax[5].set_title("Tensor: Sender Cell Type Factor")
    ax[6].set_title("Tensor: Receiver Cell Type Factor")
    ax[7].set_title("Tensor: Ligand-Receptor Factor")
    rotate_yaxis(ax[5], rotation=0)
    rotate_yaxis(ax[6], rotation=0)
    
    # Save 2 differnt X_filtered objects to compare FMS between the two different tensor decompositions
    
    
    calculateFMS_all_modes(X_filtered, X_filtered_tensor)

    return f


def load_tensor(filename, backend=None, device=None):
    '''Imports a communication tensor that could be used
    with Tensor-cell2cell.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing a communication tensor
        that was previously saved by using pickle.

    backend : str, default=None
        Backend that TensorLy will use to perform calculations
        on this tensor. When None, the default backend used is
        the currently active backend, usually is ('numpy'). Options are:
        {'cupy', 'jax', 'mxnet', 'numpy', 'pytorch', 'tensorflow'}

    device : str, default=None
        Device to use when backend allows using multiple devices. Options are:
         {'cpu', 'cuda:0', None}

    Returns
    -------
    interaction_tensor : cell2cell.tensor.BaseTensor
        A communication tensor generated with any of the tensor class in
        cell2cell.tensor.
    '''
    interaction_tensor = load_variable_with_pickle(filename)
    if 'tl' not in globals():
        import tensorly as tl
    if backend is not None:
        tl.set_backend(backend)
    
    if device is None:
        interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor)
    else:
        if tl.get_backend() in ['pytorch', 'tensorflow']:
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor, device=device)
        else:
            interaction_tensor.tensor = tl.tensor(interaction_tensor.tensor)
    
    def safe_convert_attribute(attr_name, default_value=None):
        if hasattr(interaction_tensor, attr_name):
            attr_value = getattr(interaction_tensor, attr_name)
            if attr_value is not None:
                if device is None:
                    return tl.tensor(attr_value)
                elif tl.get_backend() in ['pytorch', 'tensorflow']:
                    return tl.tensor(attr_value, device=device)
                else:
                    return tl.tensor(attr_value)
        return default_value
    
    interaction_tensor.loc_nans = safe_convert_attribute('loc_nans', None)
    interaction_tensor.loc_zeros = safe_convert_attribute('loc_zeros', None)
    interaction_tensor.mask = safe_convert_attribute('mask', None)
    
    return interaction_tensor


def load_variable_with_pickle(filename):
    '''Imports a large size variable stored in a file previously
    exported with pickle.

    Parameters
    ----------
    filename : str
        Absolute path to a file storing a python variable that
        was previously created by using pickle.

    Returns
    -------
    variable : a python variable
        The variable of interest.
    '''

    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    variable = pickle.loads(bytes_in)
    return variable


def calculateFMS_all_modes(A, B):
    """Calculates and prints FMS between 2 factors for all modes."""
    # Assume weights and factors are stored in .uns and .varm as in your code
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]] if "Pf2_D" in A.uns else [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]] if "Pf2_D" in B.uns else [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    score = fms(A_CP, B_CP, consider_weights=True)
    print(f"FMS w/Weights: {score}")
    
    score = fms(A_CP, B_CP, consider_weights=False)
    print(f"FMS w/o Weights: {score}")