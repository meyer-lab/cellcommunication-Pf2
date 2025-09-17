"""
Figure A2: FMS across CPD ranks (only) for COVID-19
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
from ..ccc_rise import calc_communication_score
from .common import getSetup, subplotLabel
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)
    rise_rank = 35

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
        projected_matrices, gene_names=gene_names, lr_pairs=lr_pairs,
        complex_sep="&"
    )
    print(np.shape(interaction_tensor))

    rank_list = list(range(1, 15, 2))
    rank_list = list(range(1, 4, 2))
    runs = 3
    runs = 1
    fms_list = []
    r2xLists = []
    for i in range(0, runs, 1):
        scores = []
        r2x_scores = []
        for j in rank_list:
            print(f"Run {i+1}, Rank {j}")
            boot_tensor = resample_tensor(interaction_tensor)
            cp_weights, cp_factors = parafac(
                tensor=interaction_tensor,
                rank=j,
                n_iter_max=1000,
                init="random",  # Use SVD initialization
                normalize_factors=True,
            )
            r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
            cp_boot_weights, cp_boot_factors = parafac(
                tensor=boot_tensor,
                rank=j,
                n_iter_max=1000,
                init="random",  # Use SVD initialization
                normalize_factors=True,
            )
            fms_score = calculateFMS(cp_weights, cp_factors, cp_boot_weights, cp_boot_factors)
            scores.append(fms_score)
            r2x_scores.append(r2x)
        # Save fms/r2x scores per rank
        fms_list.append(scores)
        r2xLists.append(r2x_scores)
        
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
    r2xList_df = []
    for sublist in r2xLists:
        r2xList_df += sublist
        
    df = pd.DataFrame(
        {"Run": runsList_df, "Component": ranksList_df, "FMS": fmsList_df, "R2X": r2xList_df}
    )

    sns.lineplot(data=df, x="Component", y="FMS", ax=ax[0], label="FMS")
    ax[0].set_ylim(0, 1)

    sns.lineplot(data=df, x="Component", y="R2X", ax=ax[1], color="orange", label="R2X")
    ax[1].set_ylim(0, np.max(df["R2X"]) + 0.02)

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
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=3)  # type: ignore


def resample_tensor(interaction_tensors):
    """Bootstrap tensor by resampling last dimension"""
    indices = np.random.randint(0, interaction_tensors.shape[-1], size=interaction_tensors.shape[-1])
    return interaction_tensors[..., indices]


def calculate_r2x(cp_weights, cp_factors, interaction_tensor):
    """Calculate R2X for the CP decomposition of the interaction tensor"""
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensor**2)
    error = np.sum((interaction_tensor - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return final_R2X