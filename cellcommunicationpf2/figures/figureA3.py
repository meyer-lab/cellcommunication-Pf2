"""
Figure A1: FMS across RISE ranks (only) for COVID-19
"""

import anndata
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from ..utils import resample
from parafac2.parafac2 import parafac2_nd, anndata_to_list
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs
)
from ..cc_pf2 import calc_communication_score
from .common import getSetup, subplotLabel
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)
    rise_rank = 5

    pf2_out, rise_rank = parafac2_nd(
            X_filtered, rank=rise_rank, n_iter_max=1000, tol=1e-9
        )
    _, _, projections = pf2_out

    # Project matrices
    X_list = anndata_to_list(X_filtered)
    projected_matrices = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        # Convert tensor to NumPy
        tensor_np = tensor.get()

        projected_matrices.append(proj.T @ tensor_np)

    # Calculate cell-cell communication scores
    gene_names = list(X_filtered.var_names)
    interaction_tensors, filtered_lr_pairs = calc_communication_score(
        projected_matrices, gene_names=gene_names
    )
    print(np.shape(interaction_tensors))
    
    

    return f


