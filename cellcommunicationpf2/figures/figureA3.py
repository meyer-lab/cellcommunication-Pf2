"""
Figure A3: FMS across CPD ranks (only) for COVID-19
"""

import pandas as pd
import seaborn as sns
import numpy as np
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
from tensorly.decomposition import parafac


def makeFigure():
    ax, f = getSetup((3, 3), (1, 1))
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
    interaction_tensor, _ = calc_communication_score(
        projected_matrices, gene_names=gene_names, lr_pairs=lr_pairs
    )
    print(np.shape(interaction_tensor))

    rank_list = list(range(1, 11, 2))
    runs = 3
    fms_list = []
    for i in range(0, runs, 1):
        scores = []
        for j in rank_list:
            boot_tensor = resample_tensor(interaction_tensor)
            cp_weights, cp_factors = parafac(
                tensor=interaction_tensor,
                rank=j,
                n_iter_max=1000,
                init="svd",  # Use SVD initialization
                normalize_factors=True,
            )
            cp_boot_weights, cp_boot_factors = parafac(
                tensor=boot_tensor,
                rank=j,
                n_iter_max=1000,
                init="svd",  # Use SVD initialization
                normalize_factors=True,
            )
            fms_score = calculateFMS(cp_weights, cp_factors, cp_boot_weights, cp_boot_factors)
            scores.append(fms_score)
        # Save fms scores per rank
        fms_list.append(scores)
        
        runsList_df = []
    for i in range(0, runs):
        for _j in range(0, len(rank_list)):
            runsList_df.append(i)
    ranksList_df = []
    for _i in range(0, runs):
        for j in range(0, len(rank_list)):
            ranksList_df.append(rank_list[j])
    fmsList_df = []
    for sublist in fms_list:
        fmsList_df += sublist
        
    df = pd.DataFrame(
        {"Run": runsList_df, "Component": ranksList_df, "FMS": fmsList_df}
    )

    sns.lineplot(data=df, x="Component", y="FMS", ax=ax[0], label="FMS")
    ax[0].set_ylim(0, 1)

    
    return f


def calculateFMS(weightsA, factorsA, weightsB, factorsB):
    """Calculates FMS between 2 factors"""
    A_CP = CPTensor(
        (
            weightsA,
            factorsA,
        )
    )
    B_CP = CPTensor(
        (
            weightsB,
            factorsB,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=(1, 2))  # type: ignore


def resample_tensor(interaction_tensors):
    """Bootstrap tensor by resampling last dimension"""
    indices = np.random.randint(0, interaction_tensors.shape[-1], size=interaction_tensors.shape[-1])
    return interaction_tensors[..., indices]