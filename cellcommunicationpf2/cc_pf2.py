import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import anndata_to_list, parafac2_nd
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, cp_to_tensor
from tensorly.decomposition import parafac

from .ccc import build_context_ccc_tensor
from .import_data import import_ligand_receptor_pairs


def calc_communication_score(
    projected_matrices: list[np.ndarray],
    gene_names: list[str] = None,
    lr_pairs: pd.DataFrame = None,
) -> np.ndarray:
    """
    Calculate cell-cell communication scores using build_context_ccc_tensor
    from Tensor Cell2Cell for all conditions at once.

    Parameters:
    -----------
    projected_matrices : list[np.ndarray]
        List of matrices of shape (rank, genes) representing projected cell expressions
        across different conditions
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns

    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, rank, rank, n_lr_pairs)
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(projected_matrices[0].shape[1])]

    # Convert matrices to DataFrames (genes as rows, cells as columns)
    rnaseq_matrices = []
    for matrix in projected_matrices:
        df = pd.DataFrame(
            matrix.T,
            index=gene_names,
            columns=[f"rank_{j}" for j in range(matrix.shape[0])],
        )
        rnaseq_matrices.append(df)

    # Rename columns to match Cell2Cell convention
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        lr_pairs_renamed = lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
    else:
        lr_pairs_renamed = lr_pairs

    # Generate communication tensor for all contexts
    tensors, _, _, _, _ = build_context_ccc_tensor(
        rnaseq_matrices=rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score="expression_product",
        complex_sep=None,
        upper_letter_comparison=False,
        interaction_columns=("A", "B"),
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=False,
    )

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, rank, rank)
    # To:   (context, rank, rank, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor


def cc_pf2(
    adata: anndata.AnnData,
    rise_rank: int,
    n_iter_max: int,
    tol: float,
    cp_rank: int | None = None,
    random_state: int | None = None,
) -> tuple[tuple, float]:
    """
    Redesigned cell-cell communication model using initial PARAFAC2
    followed by CP decomposition.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with cells x genes expression data
    rank : int
        Rank of the decomposition
    n_iter_max : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple[tuple, float]
        (((cp_weights, cp_factors), projections), final_R2X)
    """
    gene_names = list(adata.var_names)
    X_list = anndata_to_list(adata)

    # PARAFAC2 decomposition
    pf2_output, pf2_r2x = parafac2_nd(
        adata, rank=rise_rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )
    _, _, projections = pf2_output

    # Project matrices
    projected_matrices = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        # Convert tensor to NumPy
        tensor_np = tensor.get()

        projected_matrices.append(proj.T @ tensor_np)

    # Calculate cell-cell communication scores
    interaction_tensors = calc_communication_score(
        projected_matrices, gene_names=gene_names
    )

    cp_rank = cp_rank if cp_rank is not None else rise_rank

    # Print shape of interaction tensors
    print(f"Interaction tensors shape: {interaction_tensors.shape}")

    # CP decomposition with explicit random initialization
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=None,
        init="random",  # Use random initialization
        normalize_factors=False,
        random_state=random_state,
    )

    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return ((cp_weights, cp_factors), projections), final_R2X


def standardize_cc_pf2(
    factors: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Standardize CP factors and projections for better interpretability.
    This function expects all inputs to be NumPy arrays on the CPU.

    Parameters
    ----------
    factors : list[np.ndarray]
        CP factors from the decomposition.
    weights : np.ndarray, optional
        Component weights from the CP decomposition. If None, they are initialized to ones.

    Returns
    -------
    tuple
        (weights, factors) after standardization.
    """
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    if weights is not None:
        weights = weights[gini_idx]

    weights, factors = cp_flip_sign(cp_normalize((weights, factors)), mode=1)

    return weights, factors


def store_cc_pf2(X: anndata.AnnData, parafac2_output: tuple):
    """
    Store CC-PF2 results into an AnnData object.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData object to store results in.
    parafac2_output : tuple
        The standardized output from `standardize_cc_pf2`: (weights, factors, projections).

    Returns
    -------
    anndata.AnnData
        Updated AnnData with stored results.
    """
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
