import anndata
import numpy as np
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from .ccc_rise import ccc_rise, standardize_cc_pf2
from .import_data import add_cond_idxs
import pandas as pd
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2, anndata_to_list
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from .ccc_rise import calc_communication_score


def resample_tensor(interaction_tensors):
    """Bootstrap tensor by resampling last dimension"""
    indices = np.random.randint(
        0, interaction_tensors.shape[-1], size=interaction_tensors.shape[-1]
    )
    return interaction_tensors[..., indices]


def rise_store_r2x(
    X: anndata.AnnData,
    rank: int,
    n_iter_max: int,
    tolerance: float,
    random_state: int = None,
):
    """Runs RISE and stores the results."""
    pf2_out, r2x = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=n_iter_max
    )
    X = store_pf2(X, pf2_out)

    return X, r2x


def calculate_fms_cpd(weightsA, factorsA, weightsB, factorsB):
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


def calculate_fms_rise(A: anndata.AnnData, B: anndata.AnnData):
    """Calculates FMS between 2 factors"""
    factors = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    A_CP = CPTensor(
        (
            A.uns["Pf2_weights"],
            factors,
        )
    )

    factors = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    B_CP = CPTensor(
        (
            B.uns["Pf2_weights"],
            factors,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=1)  # type: ignore


def calculate_r2x(cp_weights, cp_factors, interaction_tensor):
    """Calculate R2X for the CP decomposition of the interaction tensor"""
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensor**2)
    error = np.sum((interaction_tensor - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return final_R2X


def run_ccc_rise_workflow(
    adata: anndata.AnnData,
    rise_rank: int,
    lr_pairs: pd.DataFrame,
    cp_rank: int | None = None,
    condition_column: str = "sample",
    n_iter_max: int = 100,
    tol: float = 1e-3,
    random_state: int | None = None,
    complex_sep: str = None,
    doEmbedding: bool = True,
    svd_init: str = "svd",
) -> tuple[anndata.AnnData, float]:
    """
    Executes the complete CCC-RISE workflow: decomposition, standardization,
    condition factor correction, and result storage.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with expression data.
    rank : int
        The rank for the decomposition.
    lr_pairs : pd.DataFrame
        The ligand-receptor pairs used in the decomposition.
    cp_rank : int, optional
        The rank for the final CP decomposition. If None, defaults to `rank`.
    condition_column : str, default="sample"
        The column in `adata.obs` that defines the conditions.
    n_iter_max : int
        Maximum number of iterations for the decomposition.
    tol : float
        Convergence tolerance for the decomposition.
    random_state : int | None
        Random seed for reproducibility.
    complex_sep : str, optional
        Separator for complexed ligand-receptor pairs. If None, complexes 
        are not processed. Default is None.
    doEmbedding : bool, default=True
        Whether to perform dimensionality reduction embedding (PaCMAP).
    svd_init : str, default="svd"
        Initialization method for the decomposition. Options: "svd", "random".

    Returns
    -------
    tuple[anndata.AnnData, float]
        A tuple containing the updated AnnData object with stored results
        and the R2X value of the decomposition.

    Examples
    --------
    >>> import pandas as pd
    >>> lr_pairs = pd.DataFrame({
    ...     'ligand': ['L1', 'L2'],
    ...     'receptor': ['R1', 'R2'],
    ...     'interaction_symbol': ['L1-R1', 'L2-R2']
    ... })
    >>> adata, r2x = run_ccc_rise_workflow(adata, 10, lr_pairs)

    See Also
    --------
    ccc_rise : Core function for CCC-RISE decomposition
    """
    # Ensure condition indices are present before running the model
    adata = add_cond_idxs(adata, condition_column)

    # 1. Run the CC-RISE decomposition
    results, r2x, filtered_lr_pairs = ccc_rise(
        adata,
        rise_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        cp_rank=cp_rank,
        random_state=random_state,
        complex_sep=complex_sep,
        lr_pairs=lr_pairs,
        svd_init=svd_init,
    )
    (cp_weights, factors), projections = results

    # 2. Standardize the factors for interpretability
    weights, factors = standardize_cc_pf2(cp_weights, factors)

    # Store factors in AnnData object for easy access by plotting functions
    adata.uns["A"] = factors[0]  # Condition factor
    adata.uns["B"] = factors[1]  # Sender cells factor
    adata.uns["C"] = factors[2]  # Receiver cells factor
    adata.uns["D"] = factors[3]  # LR pairs factor

    # Store the LR pairs and R2X
    adata.uns["lr_pairs"] = filtered_lr_pairs["interaction_symbol"].values
    adata.uns["r2x"] = r2x
    adata.uns["weights"] = weights
    sg_index = adata.obs["condition_unique_idxs"]
    adata.obsm["projections"] = np.zeros((adata.shape[0], rise_rank))
    for i, p in enumerate(projections):
        adata.obsm["projections"][sg_index == i, :] = p
    adata.obsm["sc_B"] = adata.obsm["projections"] @ factors[1]
    adata.obsm["rc_C"] = adata.obsm["projections"] @ factors[2]

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        adata.obsm["PaCMAP"] = pcm.fit_transform(adata.obsm["projections"])

    return adata, r2x


def calculate_interaction_tensor(
    X_filtered: anndata.AnnData, 
    lr_pairs: pd.DataFrame, 
    rise_rank: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Construct interaction tensor from expression data using RISE and 
    ligand-receptor communication scores.

    This function is a convenience wrapper that combines RISE decomposition
    with communication score calculation in a single step.

    Parameters
    ----------
    X_filtered : anndata.AnnData
        Preprocessed annotated data object with expression data in `.X`.
        Should have condition annotations in `.obs`.
    lr_pairs : pd.DataFrame
        Ligand-receptor pair annotations with 'ligand' and 'receptor' columns.
    rise_rank : int
        Rank for PARAFAC2 decomposition. Determines dimensionality of the
        latent space for cell expression patterns.

    Returns
    -------
    interaction_tensor : np.ndarray
        4D tensor of shape (n_conditions, rise_rank, rise_rank, n_lr_pairs)
        containing computed communication scores.
    projections : list of np.ndarray
        List of projection matrices, one per condition, each of shape
        (n_cells_in_condition, rise_rank). Maps cells to cell eigen-states.

    Examples
    --------
    >>> from cellcommunicationpf2 import add_cond_idxs
    >>> X = add_cond_idxs(adata, "sample")
    >>> tensor, projs = calculate_interaction_tensor(X, lr_pairs, rise_rank=35)
    >>> print(f"Tensor shape: {tensor.shape}")
    >>> print(f"Number of conditions: {len(projs)}")

    See Also
    --------
    calc_communication_score : Communication score calculation
    parafac2_nd : PARAFAC2 decomposition
    run_ccc_rise_workflow : Full analysis workflow
    """
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
        projected_matrices, gene_names=gene_names, lr_pairs=lr_pairs, complex_sep="&"
    )

    return interaction_tensor


def run_fms_r2x_analysis(
    interaction_tensor: np.ndarray,
    rank_list: list[int] = None,
    runs: int = 1,
    svd_init: str = "svd",
) -> pd.DataFrame:
    """
    Evaluate CP decomposition stability across different ranks using bootstrapping.

    This analysis helps determine the optimal rank for CP decomposition by
    computing two key metrics:
    
    1. **FMS (Factor Match Score)**: Measures similarity between factors from
       original data vs. bootstrapped data. Higher FMS indicates more stable
       decomposition.
    2. **R²X**: Variance explained by the decomposition. Indicates goodness of fit.

    Parameters
    ----------
    interaction_tensor : np.ndarray
        4D interaction tensor of shape (n_conditions, n_sender_eigen_states, 
        n_receiver_eigen_states, n_lr_pairs) to decompose and analyze.
    rank_list : list of int, optional
        List of ranks to evaluate. If None, defaults to range(1, 4, 2).
        Recommended to test multiple ranks, e.g., [2, 4, 6, 8, 10, 12].
        Default is None.
    runs : int, default=1
        Number of bootstrap iterations per rank. Higher values give more reliable
        stability estimates but increase computation time.
    svd_init : {'svd', 'random'}, default='svd'
        Initialization method for CP decomposition:
        
        - 'svd': SVD-based (deterministic, recommended)
        - 'random': Random initialization

    Returns
    -------
    results_df : pd.DataFrame
        Results with columns:
        
        - 'Run': Bootstrap run number (0 to runs-1)
        - 'Component': Rank being evaluated
        - 'FMS': Factor Match Score (0 to 1, higher is better)
        - 'R2X': Variance explained (0 to 1, higher is better)

    Examples
    --------
    >>> from cellcommunicationpf2.figures import plot_fms_r2x
    >>> rank_list = [2, 4, 6, 8, 10, 12]
    >>> results = run_fms_r2x_analysis(
    ...     interaction_tensor, 
    ...     rank_list=rank_list, 
    ...     runs=10
    ... )
    >>> # Plot results
    >>> import seaborn as sns
    >>> sns.lineplot(data=results, x='Component', y='FMS')
    >>> sns.lineplot(data=results, x='Component', y='R2X')

    See Also
    --------
    run_fms_r2x_data_percentage_analysis : Stability vs. data size
    calculate_fms_cpd : FMS calculation between two decompositions
    factor_match_score : Underlying FMS implementation (tlviz)
    """
    if rank_list is None:
        rank_list = list(range(1, 4, 2))

    fms_list = []
    r2xLists = []

    for i in range(runs):
        scores = []
        r2x_scores = []
        for j in rank_list:
            print(f"Run {i + 1}, Rank {j}")
            boot_tensor = resample_tensor(interaction_tensor)
            cp_weights, cp_factors = parafac(
                tensor=interaction_tensor,
                rank=j,
                n_iter_max=1000,
                init=svd_init,
                normalize_factors=True,
            )
            r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
            cp_boot_weights, cp_boot_factors = parafac(
                tensor=boot_tensor,
                rank=j,
                n_iter_max=1000,
                init=svd_init,
                normalize_factors=True,
            )
            fms_score = calculate_fms_cpd(
                cp_weights, cp_factors, cp_boot_weights, cp_boot_factors
            )
            scores.append(fms_score)
            r2x_scores.append(r2x)
        # Save fms/r2x scores per rank
        fms_list.append(scores)
        r2xLists.append(r2x_scores)

    # Convert to DataFrame format
    runsList_df = []
    for i in range(runs):
        for _j in range(len(rank_list)):
            runsList_df.append(i)

    ranksList_df = []
    for _i in range(runs):
        for j in range(len(rank_list)):
            ranksList_df.append(rank_list[j])

    fmsList_df = []
    for sublist in fms_list:
        fmsList_df += sublist

    r2xList_df = []
    for sublist in r2xLists:
        r2xList_df += sublist

    df = pd.DataFrame(
        {
            "Run": runsList_df,
            "Component": ranksList_df,
            "FMS": fmsList_df,
            "R2X": r2xList_df,
        }
    )

    return df


def run_fms_r2x_data_percentage_analysis(
    X_filtered: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    rise_rank: int,
    cp_rank: int,
    percentage_list: list[int] = None,
    runs: int = 1,
    svd_init: str = "svd",
) -> pd.DataFrame:
    if percentage_list is None:
        percentage_list = list(range(100, 45, -5))

    # 0) Reference on FULL data (no bootstrapping here)
    full_tensor = calculate_interaction_tensor(
        X_filtered, lr_pairs, rise_rank=rise_rank
    )
    w_full, f_full = parafac(
        tensor=full_tensor,
        rank=cp_rank,
        n_iter_max=1000,
        init=svd_init,
        normalize_factors=True,
    )

    rows = []
    n_cells = X_filtered.n_obs
    for run_idx in range(runs):
        for pct in percentage_list:
            # 1) Subsample WITHOUT replacement
            if pct < 100:
                n_keep = int(n_cells * pct / 100)
                keep_idx = np.random.choice(n_cells, n_keep, replace=False)
                X_sub = X_filtered[keep_idx]
            else:
                X_sub = X_filtered

            # 2) Tensor on subsample
            sub_tensor = calculate_interaction_tensor(
                X_sub, lr_pairs, rise_rank=rise_rank
            )

            # 3) CP on subsample
            w_sub, f_sub = parafac(
                tensor=sub_tensor,
                rank=cp_rank,
                n_iter_max=1000,
                init=svd_init,
                normalize_factors=True,
            )

            # 4) R²X on subsample
            r2x_sub = calculate_r2x(w_sub, f_sub, sub_tensor)

            # 5) FMS vs FULL reference (NOT vs bootstrap)
            fms_vs_full = calculate_fms_cpd(w_full, f_full, w_sub, f_sub)

            rows.append(
                {
                    "Run": run_idx,
                    "Data_Percentage": pct,
                    "FMS": fms_vs_full,
                    "R2X": r2x_sub,
                }
            )

    return pd.DataFrame(rows)
