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

    cond_names = ["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"]
    celltypes = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    total_df = []
    for i, cond_name in enumerate(cond_names):
        df = pd.read_csv(f"{cond_name}.csv")
        df.set_index(df.columns[0], inplace=True)
        # Only keep these cell types in columns of the dataframe
        df = df[celltypes]
        total_df.append(df.fillna(0))
    
    print(total_df[0]) 
        
    
    lr_pairs = import_ligand_receptor_pairs()
    # Override ligand and receptor columns using interaction_name_2
    if 'interaction_name_2' in lr_pairs.columns:
        lr_pairs['ligand'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[0].upper())
        lr_pairs['receptor'] = lr_pairs['interaction_name_2'].apply(lambda x: x.split(' - ')[1].upper().replace('(', '').replace(')', '').replace('+', '&'))
    # Also update interaction_symbol if present
    if 'interaction_symbol' in lr_pairs.columns:
        lr_pairs['interaction_symbol'] = lr_pairs['interaction_symbol'].str.upper().str.replace('_', '&')
        
    tc2c_tensor = load_tensor("Tensor-BALF.pkl")

    valid_tc2c_pairs = set(tc2c_tensor.order_names[1])
    lr_pairs_filtered = lr_pairs[lr_pairs['interaction_symbol'].isin(valid_tc2c_pairs)].reset_index(drop=True)
    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(total_df, lr_pairs=lr_pairs_filtered, complex_sep="&")

    print(np.shape(interaction_tensor))
    weights, cpd_factors, _ = pseudobulk_nncp_decomposition(interaction_tensor, cp_rank=15, n_iter_max=100000, tol=1e-11, random_state=0)
    # Confirm cpd factors are only positive
    for factor in cpd_factors:
        assert np.all(factor >= 0), "CPD factors contain negative values"


    print(cpd_factors[1].shape)
    X = import_balf_covid(gene_threshold=0)
    X.uns["Pf2_A"] = cpd_factors[0]  # Condition factor
    X.uns["Pf2_B"] = cpd_factors[1]  # Sender cell types factor
    X.uns["Pf2_C"] = cpd_factors[2]  # Receiver cell types factor
    X.uns["Pf2_D"] = cpd_factors[3]  # LR pairs factor
    X.uns["Pf2_lr_pairs"] = filtered_lr_pairs  # LR pairs

    X.uns["Pf2_weights"] = weights

    plot_condition_factors(
        data=X,
        ax=ax[0],
        cond="sample",
        # cond_group_labels=sample_to_group,
        # group_cond=True,
        yt = ["C51", "C52", "C100", "C141", "C142", "C144", "C145", "C143", "C146", "C148", "C149", "C152"]
    )


    plot_eigenstate_factors(
        data=X,
        ax=ax[1],
        factor_type="Pf2_B",
    )
    tc2c_celltype_names = ["B", "Epithelial", "Macrophages", "NK", "T", "mDC"]
    # ax[1].set_yticks(np.arange(len(tc2c_celltype_names)))
    ax[1].set_yticklabels(tc2c_celltype_names)

    plot_eigenstate_factors(
        data=X,
        ax=ax[2],
        factor_type="Pf2_C",
    )
    # ax[2].set_yticks(np.arange(len(tc2c_celltype_names)))
    ax[2].set_yticklabels(tc2c_celltype_names)

    plot_lr_factors(
        data=X,
        ax=ax[3],
        weight=0.15
    )

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
