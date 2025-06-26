import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, cp_to_tensor
from tensorly.decomposition import parafac


def temp_calc_communication_score(ces_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate cell-cell interaction scores for a given CES matrix.
    This function is a temporary placeholder and should be replaced with
    the actual implementation of calc_communication_score.
    """
    # Placeholder for the actual calculation logic. Will currently just transform the
    # matrix to a 3D tensor by filling the third dimension with random values.
    # Replace this with the actual implementation through liana.
    rank = ces_matrix.shape[0]
    genes = ces_matrix.shape[1]
    # Create a random tensor of shape (rank, rank, genes) to simulate the interaction
    interaction_tensor = np.random.rand(rank, rank, genes)
    return interaction_tensor


def cc_pf2_redesigned(
    X_list: list[np.ndarray],
    rank: int,
    n_iter_max: int,
    tol: float,
    random_state: int | None = None,
) -> tuple[tuple, float]:
    """
    Redesigned cell-cell communication model using initial PARAFAC2
    followed by CP decomposition
    """

    # Calculate total number of cells across all conditions
    total_cells = sum(x.shape[0] for x in X_list)

    # Concatenate all matrices to create the full data matrix
    X_full = np.vstack(X_list)

    # Create condition indices for each cell
    condition_idxs = []
    for i, x in enumerate(X_list):
        condition_idxs.extend([i] * x.shape[0])

    # Create observation dataframe
    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(total_cells)])
    obs_df["condition_unique_idxs"] = condition_idxs

    # Create the AnnData object
    adata = anndata.AnnData(X=X_full, obs=obs_df)

    # Call parafac2_nd with our constructed AnnData
    pf2_output, _ = parafac2_nd(
        adata, rank=rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )

    # Unpack results
    _, _, projections = pf2_output

    # Step 2: Project each matrix down to standardized dimension
    projected_tensors = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        projected_tensors.append(proj.T @ tensor)  # (rank x genes)

    # Step 3: Calculate cell-cell interaction scores for each sample
    interaction_tensors = []
    for i, ces_matrix in enumerate(projected_tensors):
        # This creates (rank x rank x genes) tensors from (rank x genes) matrices
        interaction_tensor = temp_calc_communication_score(ces_matrix)
        interaction_tensors.append(interaction_tensor)

    interaction_tensors = np.stack(interaction_tensors)

    # Step 4: Run standard CP decomposition on the interaction tensors
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        rank,
        n_iter_max=20,
        tol=None,
        normalize_factors=False,
    )

    # Calculate final R2X
    reconstructed = cp_to_tensor((cp_weights, cp_factors))

    # Calculate total variance and error
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)

    # Calculate R2X (1 - normalized error)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return (cp_factors, projections), final_R2X


def standardize_cc_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    for i in [1, 2]:
        # Order eigen-cells to maximize the diagonal of B/C
        _, col_ind = linear_sum_assignment(np.abs(factors[i].T), maximize=True)
        factors[i] = factors[i][col_ind, :]
        projections = [p[:, col_ind] for p in projections]

        # Flip the sign based on B/C
        signn = np.sign(np.diag(factors[i]))
        factors[i] *= signn[:, np.newaxis]
        projections = [p * signn for p in projections]

    return weights, factors, projections


def store_cc_pf2(X: anndata.AnnData, parafac2_output: tuple):
    """Store the Pf2 results into the anndata object."""

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.uns["Pf2_C"], X.varm["Pf2_D"] = parafac2_output[1]
    projections = parafac2_output[2]

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
    cell_cell_indices = np.zeros(n_pairs, dtype=int)

    current_idx = 0
    # Process each sample separately
    samples = X.obs["sample"].unique()
    for k in range(len(samples)):
        sample_proj = projections[k]
        n_cells = sample_proj.shape[0]

        # Generate all cell pairs for this sample
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:  # Skip self-interactions
                    # Calculate projection products and store at current index
                    proj_scores[current_idx, :] = sample_proj[i] * sample_proj[j]
                    cell_cell_indices.append(k)
                    current_idx += 1

    X.obsm["Pf2_cell_cell_projections"] = proj_scores
    X.obsm["Pf2_cell_cell_weighted_projections"] = proj_scores @ X.uns["Pf2_B"]
    X.obsm["Pf2_cell_cell_condition"] = cell_cell_indices

    return X
