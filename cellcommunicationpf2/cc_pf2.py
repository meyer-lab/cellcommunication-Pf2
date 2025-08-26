import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import anndata_to_list, parafac2_nd
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, cp_to_tensor
from tensorly.decomposition import parafac, non_negative_parafac

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
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
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
    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
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

    # Filter the original lr_pairs to match the pairs used in the tensor
    lr_pair_names = lr_pairs["ligand"] + "^" + lr_pairs["receptor"]
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, rank, rank)
    # To:   (context, rank, rank, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def cc_pf2(
    adata: anndata.AnnData,
    rise_rank: int,
    n_iter_max: int,
    tol: float,
    cp_rank: int | None = None,
    random_state: int | None = None,
) -> tuple[tuple, float, pd.DataFrame]:
    """
    Perform PARAFAC2 decomposition on an AnnData object, followed by
    CP decomposition on the resulting interaction tensor.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object with expression data in `.X`
    rise_rank : int
        Rank for the PARAFAC2 decomposition (RISE)
    n_iter_max : int
        Maximum number of iterations for PARAFAC2
    tol : float
        Convergence tolerance for PARAFAC2
    cp_rank : int, optional
        Rank for the CP decomposition. If None, defaults to `rise_rank`.
    random_state : int, optional
        Seed for reproducibility

    Returns:
    --------
    tuple
        A tuple containing:
        - A nested tuple with CP results and projections: ((cp_weights, cp_factors), projections)
        - The R2X (variance explained) of the CP decomposition on the interaction tensor.
        - The filtered ligand-receptor pairs DataFrame used in the analysis.
    """
    gene_names = list(adata.var_names)
    X_list = anndata_to_list(adata)

    # PARAFAC2 decomposition
    pf2_output, _ = parafac2_nd(
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
    interaction_tensors, filtered_lr_pairs = calc_communication_score(
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

    # Calculate R2X for the CP decomposition of the interaction tensor
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return ((cp_weights, cp_factors), projections), final_R2X, filtered_lr_pairs


def calc_communication_score_pseudobulk(
    pseudobulk_matrices_df: list[pd.DataFrame],
    gene_names: list[str] = None,
    lr_pairs: pd.DataFrame = None,
) -> np.ndarray:
    """
    Calculate cell-cell communication scores for pseudobulk 

    Parameters:
    -----------
    pseudobulk_matrices_df : list[pd.DataFrame]
        List of dataframes of shape (genes, groupby)
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns

    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, sender, receiver, n_lr_pairs)
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(pseudobulk_matrices_df[0].shape[1])]

    # Rename columns to match Cell2Cell convention
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        lr_pairs_renamed = lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
    else:
        lr_pairs_renamed = lr_pairs

    # Generate communication tensor for all contexts
    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=pseudobulk_matrices_df,
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

    # Filter the original lr_pairs to match the pairs used in the tensor
    lr_pair_names = lr_pairs["ligand"] + "^" + lr_pairs["receptor"]
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, sender, receiver)
    # To:   (context, sender, receiver, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def pseudobulk_nncp_decomposition(
    interaction_tensors: np.ndarray,
    cp_rank: int,
    n_iter_max: int,
    tol: float | None = None,
    random_state: int | None = None,
) -> tuple[tuple[np.ndarray, list[np.ndarray]], float, pd.DataFrame]:
    """
    Perform non-negative CP decomposition on the pseudobulk interaction tensors.

    Parameters
    ----------
    interaction_tensors : np.ndarray
        The interaction tensors to decompose.
    cp_rank : int
        The rank for the CP decomposition.
    n_iter_max : int
        The maximum number of iterations for the decomposition.
    tol: float | None
        Tolerance for convergence.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[tuple[np.ndarray, list[np.ndarray]], float, pd.DataFrame]
        A tuple containing the CP decomposition results, the R2X value,
        and the filtered ligand-receptor pairs.
    """
    # Nonnegative CP decomposition
    nncp_weights, nncp_factors = non_negative_parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init="svd",  # Use SVD initialization
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    reconstructed = cp_to_tensor((nncp_weights, nncp_factors))
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)
    r2x = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return nncp_weights, nncp_factors, r2x



def standardize_cp_decomposition(
    weights: np.ndarray | None = None,
    factors: list[np.ndarray] | None = None
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Standardize CP factors for better interpretability.
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
