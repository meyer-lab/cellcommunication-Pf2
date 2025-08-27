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
from tensorly.decomposition import parafac,


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


    
    ranks = list(range(1, 11, 2))
    for i in ranks: 
        boot_tensors = bootstrap_tensor_features(interaction_tensors, n_bootstrap=1000)
        print(f"Bootstrapped tensors shape: {boot_tensors.shape}")
        tensors = [interaction_tensors, boot_tensors]
        for j in tensors: 
            cp_weights, cp_factors = parafac(
                tensor=j,
                rank=i,
                n_iter_max=1000,
                init="svd",  # Use SVD initialization
                normalize_factors=True,
            )
            
            
            

    
    
    return f



def calculateFMS(A: anndata.AnnData, B: anndata.AnnData):
    """Calculates FMS between 2 factors"""
    factors = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]]
    A_CP = CPTensor(
        (
            A.uns["Pf2_weights"],
            factors,
        )
    )

    factors = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]]
    B_CP = CPTensor(
        (
            B.uns["Pf2_weights"],
            factors,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=(1, 2))  # type: ignore


def bootstrap_tensor_features(interaction_tensors, n_bootstrap=100):
    """Bootstrap by resampling along the LR dimension"""
    n_features = interaction_tensors.shape[-1]  
    
    bootstrapped_tensors = []
    
    for i in range(n_bootstrap):
        # Resample feature indices with replacement
        boot_indices = np.random.choice(n_features, size=n_features, replace=True)
        boot_tensor = interaction_tensors[..., boot_indices]
        bootstrapped_tensors.append(boot_tensor)
    
    return np.array(bootstrapped_tensors) 

def store_cpd(X, cpd_factors):
    """Store CPD factors in AnnData object"""
    X.uns["Pf2_A"] = cpd_factors[0]  # Condition factor
    X.uns["Pf2_B"] = cpd_factors[1]  # Sender eigen-states factor
    X.uns["Pf2_C"] = cpd_factors[2]  # Receiver eigen-states factor
    X.uns["Pf2_D"] = cpd_factors[3]  # LR pairs factor
