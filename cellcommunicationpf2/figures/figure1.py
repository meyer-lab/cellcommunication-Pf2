"""
Figure 1
"""

from .common import getSetup, subplotLabel
from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
    anndata_lrp_overlap,
    add_cond_idxs,
    anndata_to_tensor
    
    
)
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment
from ..ccc import calc_communication_score
from ..ccc_pf2 import ccc_pf2
import numpy as np

def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)

    X = import_balf_covid()
    df_lrp = import_ligand_receptor_pairs()
    X, df_lrp = anndata_lrp_overlap(X, df_lrp)

    # Make smaller dataset for now
    X = X[::100]
    df_lrp = df_lrp.iloc[:20, :]

    Xccc = calc_communication_score(X, df_lrp, communication_score="expression_product")
    
    Xccc = add_cond_idxs(Xccc, "sample")
    print(Xccc)
    X_list = anndata_to_tensor(Xccc)

    assert len(X_list) == len(Xccc.obs["condition_unique_idxs"].unique())
    assert len(X_list[0].shape) == 3
    assert X_list[0].shape[0] == X_list[0].shape[1]
    assert X_list[0].shape[2] == X_list[1].shape[2] 
    
    
    dense_list = []
    for tensor in X_list:
        dense_tensor = tensor.todense()
        dense_list.append(dense_tensor)


    print(np.shape(dense_list[0]))
    # normalized_list = []
    # for tensor in dense_list:
        
    #     # tensor = tensor + 1e-10
    #     # Normalize each tensor
    #     # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    #     # count amoutn of non Nan values
    #     # print(np.isnan(tensor).sum())
    #     # print(np.isinf(tensor).sum())
    #     print(np.count_nonzero(tensor))
    #     print(np.shape(tensor))
        # normalized_list.append(tensor)

    
    output, r2x = ccc_pf2(dense_list, rank=2, n_iter_max=5, tol=1e-5)
    
    
    weights, factors, projections = standardize_pf2(output[0], output[1])

    # Store the results in the anndata object
    print(np.shape(weights))
    print(len(factors))
    print(len(projections))
    
    Xccc = store_pf2(Xccc, (weights, factors, projections))
    print(Xccc)
    

    return f



def standardize_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    print(np.shape(projections[0]))
    for i in [1, 2]:
        # Order eigen-cells to maximize the diagonal of B
        _, col_ind = linear_sum_assignment(np.abs(factors[i].T), maximize=True)
        factors[i] = factors[i][col_ind, :]
        projections = [p[:, col_ind] for p in projections]

        # Flip the sign based on B/C
        signn = np.sign(np.diag(factors[i]))
        factors[i] *= signn[:, np.newaxis]
        projections = [p * signn for p in projections]

    print(np.shape(projections[0]))
    return weights, factors, projections



def store_pf2(
    X,
    parafac2_output):
    """Store the Pf2 results into the anndata object."""

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"],  X.uns["Pf2_C"], X.varm["Pf2_D"] = parafac2_output[1]
    projections = parafac2_output[2]
    X.uns["Pf2_projections"] = projections
    
    stacked_projections = np.vstack(projections)
    
    # Create condition index vector matching stacked projections
    condition_indices = []
    for idx, proj in enumerate(projections):
        condition_indices.extend([idx] * proj.shape[0])
    
    # Store both stacked projections and their condition indices
    X.uns["Pf2_projections"] = stacked_projections
    X.uns["Pf2_weighted_projections"] = stacked_projections @ X.uns["Pf2_B"]
    X.uns["Pf2_projection_conditions"] = np.array(condition_indices)
    
    n_pairs = X.shape[0]  # Number of cell-cell pairs
    n_components = len(X.uns["Pf2_weights"])  # Number of components
    
    # Initialize projection scores matrix
    proj_scores = np.zeros((n_pairs, n_components))
   
    current_idx = 0
    
    # Process each sample separately
    samples = X.obs["sample"].unique()
    for k in range(len(samples)):
        sample_proj = projections[k]  # Get projections for this sample
        n_cells = sample_proj.shape[0]
        
        # Generate all cell pairs for this sample
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:  # Skip self-interactions
                    # Calculate projection products and store at current index
                    proj_scores[current_idx, :] = sample_proj[i] * sample_proj[j]
                    current_idx += 1
                    
    
    # Convert to DataFrame and create new layer
    X.obsm["Pf2_cell_cell_projections"] = proj_scores
    X.obsm["Pf2_cell_cell_projections"] = proj_scores @ X.uns["Pf2_B"]

    return X
