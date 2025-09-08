from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
def calculateFMS_all_modes(A, B):
    """Calculates and prints FMS between 2 factors for all modes."""
    # Assume weights and factors are stored in .uns and .varm as in your code
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]] if "Pf2_D" in A.uns else [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]] if "Pf2_D" in B.uns else [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    score = fms(A_CP, B_CP, consider_weights=True)
    print(f"FMS: {score}")
    
    score = fms(A_CP, B_CP, consider_weights=False)
    print(f"FMS: {score}")
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

import pandas as pd
import pickle
import os

def makeFigure():
    # ...existing code for main analysis loop...

    ax, f = getSetup((24, 12), (2, 4))
    subplotLabel(ax)
    
    

    df1 = pd.read_csv("C145.csv")
    gene_col = df1.columns[0]
    tensor = load_tensor("Tensor-BALF.pkl")
    # print("Loaded tensor with shape:", np.array(tensor.tensor))

    # print(tensor.order_names[1])
    

    #Import and prepare data
    print("Importing data...")
    X = import_balf_covid(gene_threshold=0)
    
    
    X.var_names = [name.upper() for name in X.var_names]
    lr_pairs = import_ligand_receptor_pairs()
    
    lta_interaction = lr_pairs[lr_pairs['interaction_symbol'].str.startswith('ITGB2')]
    print(lta_interaction)
        
    
    # Check genes names are same in both datasets (tensor and X)
    tensor_genes = set(tensor.genes)
    X_genes = set(X.var_names)
    common_genes = tensor_genes.intersection(X_genes)
    print(f"Number of common genes between tensor and X: {len(common_genes)} out of {len(tensor_genes)} tensor genes and {len(X_genes)} X genes")
    print(f"unique genes in tensor not in X: {tensor_genes - X_genes}")
    
    # See if specific genes are in X var_names
    specific_genes = ['ITGB2', 'TGFB1', 'LTA']
    for gene in specific_genes:
        if gene in X.var_names:
            print(f"{gene} is present in X var_names")
        else:
            print(f"{gene} is NOT present in X var_names")

    # Override ligand and receptor columns using interaction_name_2
    if 'interaction_name_2' in lr_pairs.columns:
        lr_pairs['ligand'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[0].upper())
        lr_pairs['receptor'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[1].upper().replace('(', '').replace(')', '').replace('+', '&'))
    # Also update interaction_symbol if present
    if 'interaction_symbol' in lr_pairs.columns:
        lr_pairs['interaction_symbol'] = lr_pairs['interaction_symbol'].str.upper().str.replace('_', '&')
    
    
    
    only_in_tensor = {
    'ITGB2^CD226', 'TGFB1^ACVR1&TGFBR1', 'LTA^LTB&LTBR', 'ITGB2^ICAM2',
    'SEMA3C^NRP1&NRP2', 'ITGB2^ICAM1', 'EBI3^IL27RA&IL6ST', 'TNFSF10^TNFRSF10B'
}

    present = []
    not_present = []
    for pair in only_in_tensor:
        if pair in lr_pairs["interaction_symbol"].values:
            present.append(pair)
        else:
            not_present.append(pair)

    print("Present in lr_pairs:", present)
    print("Not present in lr_pairs:", not_present)
    
            
    lta_interaction = lr_pairs[lr_pairs['interaction_symbol'].str.startswith('ITGB2')]
    print(lta_interaction)
        
    
    # matches_filtered = filtered_lr_pairs[filtered_lr_pairs["interaction_symbol"] == search_str]
    # print("Matches in filtered_lr_pairs:")
    # print(matches_filtered)

    # print(lr_pairs)

    cell_types = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    X = X[X.obs["celltype"].isin(cell_types)]

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
    types = ["fraction"]
    # types = ["mean", "fraction"]

    # Save X_filtered for the first decomposition
    X_filtered_first = X_filtered.copy()
    for i, t in enumerate(types):
        appended_pseudobulk = pseudobulk_X(X_filtered, condition_name=condition_column, groupby=groupby, type=t)
        valid_pairs = set(tensor.order_names[1])
        lr_pairs_filtered = lr_pairs[lr_pairs['interaction_symbol'].isin(valid_pairs)].reset_index(drop=True)
        interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(appended_pseudobulk, lr_pairs=lr_pairs_filtered, complex_sep="&")

        # print(filtered_lr_pairs)
        overlap = set(filtered_lr_pairs["interaction_symbol"].values) & set(tensor.order_names[1])
        print("Overlapping LR pairs:", len(overlap))
        print("Length of tensor order names:", len(tensor.order_names[1]))
        print("Length of filtered LR pairs:", filtered_lr_pairs.shape[0])

        only_in_tensor = set(tensor.order_names[1]) - set(filtered_lr_pairs["interaction_symbol"].values)
        print("Only in tensor_order_names:", only_in_tensor)

        weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=10, n_iter_max=10000, tol=1e-9)
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
        

    valid_tensor_pairs = set(filtered_lr_pairs['interaction_symbol'])
    tensor_lr_pairs = np.array(tensor.order_names[1])
    keep_indices = [i for i, pair in enumerate(tensor_lr_pairs) if pair in valid_tensor_pairs]
    # Filter the tensor along the LR pair mode (last mode)
    tensor_filtered = np.take(np.array(tensor.tensor), keep_indices, axis=1)
    cp_rank = 10  # You can adjust this rank as needed
    n_iter_max = 10000
    tol = 1e-9
    print("Performing non-negative CP decomposition on the loaded tensor...")
    nncp_weights, nncp_factors, r2x = pseudobulk_nncp_decomposition(
        tensor_filtered,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol
)
    for factor in nncp_factors:
        assert np.all(factor >= 0), "CPD factors contain negative values"

    X_filtered_tensor.uns["Pf2_A"] = nncp_factors[0]  # Condition factor
    X_filtered_tensor.uns["Pf2_B"] = nncp_factors[2]  # Sender cell types factor
    X_filtered_tensor.uns["Pf2_C"] = nncp_factors[3]  # Receiver cell types factor
    X_filtered_tensor.uns["Pf2_D"] = nncp_factors[1]  # LR pairs factor
    X_filtered_tensor.uns["Pf2_lr_pairs"] = filtered_lr_pairs  # LR pairs
    X_filtered_tensor.uns["Pf2_weights"] = nncp_weights
         
    plot_condition_factors(
        data=X_filtered_tensor,
        ax=ax[4],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,
    )
    plot_eigenstate_factors(
        data=X_filtered_tensor,
        ax=ax[5],
        factor_type="Pf2_B",
    )
    ax[5].set_yticklabels(groupby_names)
    plot_eigenstate_factors(
        data=X_filtered_tensor,
        ax=ax[6],
        factor_type="Pf2_C",
    )
    ax[6].set_yticklabels(groupby_names)
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

