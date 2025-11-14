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
    complex_sep: str = None,
    complex_agg_method: str = "min",
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate cell-cell communication scores using build_context_ccc_tensor.

    This function computes communication scores for all conditions simultaneously
    using projected cell expression matrices and ligand-receptor pair information.
    It leverages the Tensor Cell2Cell framework to generate interaction tensors.

    Parameters
    ----------
    projected_matrices : list of np.ndarray
        List of matrices of shape (rank, genes) representing projected cell 
        expressions across different conditions. Each matrix corresponds to one
        condition/sample/context.
    gene_names : list of str, optional
        List of gene names corresponding to columns in the matrices. If None,
        generates generic names like 'gene_0', 'gene_1', etc. Default is None.
    lr_pairs : pd.DataFrame, optional
        DataFrame containing ligand-receptor pairs with columns 'ligand' and 
        'receptor'. If None, loads default pairs via import_ligand_receptor_pairs().
        Default is None.
    complex_sep : str, optional
        Symbol separating protein subunits in multimeric complexes (e.g., '&').
        If None, complexes are not processed. Default is None.
    complex_agg_method : {'min', 'mean', 'gmean'}, default='min'
        Method to aggregate expression values for protein complexes:
        
        - 'min': Minimum expression among subunits
        - 'mean': Average expression among subunits
        - 'gmean': Geometric mean of expression among subunits
    verbose : bool, default=False
        If True, prints progress information during computation.

    Returns
    -------
    interaction_tensor : np.ndarray
        4D interaction tensor of shape (n_conditions, rank, rank, n_lr_pairs)
        containing communication scores. Dimensions represent:
        
        - axis 0: conditions/samples/contexts
        - axis 1: sender cell ranks
        - axis 2: receiver cell ranks
        - axis 3: ligand-receptor pairs
    filtered_lr_pairs : pd.DataFrame
        Filtered ligand-receptor pairs that correspond to the tensor's last
        dimension. Only includes pairs where both ligand and receptor genes
        are present in the expression data.

    Notes
    -----
    This function internally converts matrices to DataFrames with genes as rows
    and cells/ranks as columns, following the Tensor Cell2Cell convention.

    Examples
    --------
    >>> projected_mats = [np.random.rand(10, 100) for _ in range(5)]
    >>> gene_names = [f"gene_{i}" for i in range(100)]
    >>> tensor, lr_pairs = calc_communication_score(projected_mats, gene_names)
    >>> print(tensor.shape)
    (5, 10, 10, 2000)

    See Also
    --------
    ccc_rise : Main workflow function that uses this internally
    build_context_ccc_tensor : Underlying Tensor Cell2Cell function
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

    # Handle complexes if requested
    if complex_sep is not None:
        if verbose:
            print("Getting expression values for protein complexes")
        _, _, _, _, complexes = get_genes_from_complexes(
            ppi_data=lr_pairs_renamed,
            complex_sep=complex_sep,
            interaction_columns=("A", "B"),
        )
        mod_rnaseq_matrices = [
            add_complexes_to_expression(
                rnaseq, complexes, agg_method=complex_agg_method
            )
            for rnaseq in rnaseq_matrices
        ]
    else:
        mod_rnaseq_matrices = [df.copy() for df in rnaseq_matrices]

    # Generate communication tensor for all contexts
    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score="expression_product",
        complex_sep=complex_sep,
        upper_letter_comparison=False,
        interaction_columns=("A", "B"),
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
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


def ccc_rise(
    adata: anndata.AnnData,
    rise_rank: int,
    n_iter_max: int,
    tol: float,
    cp_rank: int | None = None,
    random_state: int | None = None,
    complex_sep: str = None,
    lr_pairs: pd.DataFrame = None,
    svd_init: str = "svd",
) -> tuple[tuple, float, pd.DataFrame]:
    """
    Perform RISE followed by CP decomposition.

    This is the core analytical workflow that:
    1. Applies RISE to expression data across conditions
    2. Projects cells x genes matrices onto cell eigen-state x genes space
    3. Computes cell-cell communication scores across all conditions
    4. Performs CP tensor decomposition on the resulting interaction tensor

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing expression data in `.X`. Must have
        consistent gene names across all observations and condition annotations.
    rise_rank : int
        Rank for the PARAFAC2 decomposition (RISE step). Determines the number
        of cell eigen-states to project into. Typical values range from 10-50 
        depending on complexity.
    n_iter_max : int
        Maximum number of iterations for both PARAFAC2 and CP decomposition
        algorithms.
    tol : float
        Convergence tolerance for both decomposition algorithms. Algorithm stops
        when the relative change in reconstruction error falls below this threshold.
    cp_rank : int, optional
        Rank for the final CP decomposition on the interaction tensor. If None, 
        defaults to `rise_rank`. Can be set lower than rise_rank to reduce 
        dimensionality of communication patterns while capturing more cell 
        populations. Default is None.
    random_state : int, optional
        Random seed for reproducibility of decomposition algorithms. If None,
        results may vary between runs. Default is None.
    complex_sep : str, optional
        Separator symbol for protein complexes in ligand-receptor annotations
        (e.g., '&' for "CD74&CD44"). If None, protein complexes are not handled
        specially. Default is None.
    lr_pairs : pd.DataFrame, optional
        Custom ligand-receptor pairs DataFrame with 'ligand' and 'receptor' columns.
        If None, uses default database via import_ligand_receptor_pairs(). 
        Default is None.
    svd_init : {'svd', 'random'}, default='svd'
        Initialization method for CP decomposition:
        
        - 'svd': Uses SVD-based initialization
        - 'random': Random initialization

    Returns
    -------
    results : tuple
        Nested tuple containing:
        
        - ((cp_weights, cp_factors), projections) where:
            - cp_weights : np.ndarray of shape (cp_rank,)
                Component importance weights
            - cp_factors : list of 4 np.ndarray
                [condition_factor, sender_factor, receiver_factor, lr_pair_factor]
            - projections : list of np.ndarray
                PARAFAC2 projections for each condition
    r2x : float
        Variance explained (R²X) by the CP decomposition on the interaction tensor.
        Values range from 0 to 1, where higher is better. Typical good values: >0.7.
    filtered_lr_pairs : pd.DataFrame
        Ligand-receptor pairs actually used in the analysis after filtering for
        gene availability in the expression data.

    Raises
    ------
    ValueError
        If `adata` is missing required fields or has inconsistent dimensions.

    Notes
    -----
    The RISE algorithm extends PARAFAC2 to handle irregular tensors where different
    conditions may have different numbers of cells. The resulting interaction tensor
    has regular dimensions based on the learned cell eigen-state dimension.

    The R²X metric indicates how well the CP decomposition reconstructs the
    interaction tensor.

    Examples
    --------
    Basic usage with COVID-19 BALF data:

    >>> from cellcommunicationpf2 import import_balf_covid, import_ligand_receptor_pairs
    >>> adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    >>> lr_pairs = import_ligand_receptor_pairs()
    >>> results, r2x, lr_pairs_used = ccc_rise(
    ...     adata, 
    ...     rise_rank=35, 
    ...     cp_rank=8,
    ...     n_iter_max=1000, 
    ...     tol=1e-9,
    ...     complex_sep='&',
    ...     lr_pairs=lr_pairs
    ... )
    >>> print(f"Variance explained: {r2x:.3f}")
    >>> (cp_weights, cp_factors), projections = results

    See Also
    --------
    run_ccc_rise_workflow : High-level wrapper with additional post-processing
    calc_communication_score : Communication score calculation
    standardize_cc_pf2 : Factor standardization for interpretability
    parafac2_nd : Underlying PARAFAC2 implementation
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
        projected_matrices,
        gene_names=gene_names,
        complex_sep=complex_sep,
        lr_pairs=lr_pairs,
    )

    cp_rank = cp_rank if cp_rank is not None else rise_rank

    # Print shape of interaction tensors
    print(f"Interaction tensors shape: {interaction_tensors.shape}")

    # CP decomposition with explicit random initialization
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init=svd_init,  # Use random initialization
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return ((cp_weights, cp_factors), projections), final_R2X, filtered_lr_pairs


def standardize_cc_pf2(
    weights: np.ndarray | None = None, factors: list[np.ndarray] = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Standardize CP tensor factors for improved interpretability and consistency.

    This function performs three key standardization steps:
    1. Orders components by condition variance (Gini coefficient)
    2. Normalizes factor magnitudes using cp_normalize
    3. Resolves sign ambiguity using cp_flip_sign

    Parameters
    ----------
    weights : np.ndarray, optional
        Component importance weights of shape (n_components,). If None,
        initialized to ones. Default is None.
    factors : list of np.ndarray, optional
        List of factor matrices from CP decomposition. Expected order:
        [conditions, senders, receivers, lr_pairs]. Default is None.

    Returns
    -------
    weights : np.ndarray
        Standardized component weights of shape (n_components,).
    factors : list of np.ndarray
        Standardized factor matrices in the same order as input.

    Examples
    --------
    >>> cp_weights = np.array([1.5, 2.3, 0.8])
    >>> cp_factors = [
    ...     np.random.rand(10, 3),  # conditions
    ...     np.random.rand(20, 3),  # senders
    ...     np.random.rand(20, 3),  # receivers
    ...     np.random.rand(50, 3),  # lr_pairs
    ... ]
    >>> std_weights, std_factors = standardize_cc_pf2(cp_weights, cp_factors)

    See Also
    --------
    run_ccc_rise_workflow : Workflow that uses this standardization
    """
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    if weights is not None:
        weights = weights[gini_idx]

    weights, factors = cp_flip_sign(cp_normalize((weights, factors)), mode=1)

    return weights, factors


def calc_communication_score_pseudobulk(
    pseudobulk_matrices_df: list[pd.DataFrame],
    gene_names: list[str] = None,
    lr_pairs: pd.DataFrame = None,
    complex_sep: str = "&",
    complex_agg_method: str = "min",
    verbose: bool = False,
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
    complex_sep : str, default="&"
        Symbol that separates the protein subunits in a multimeric complex
    complex_agg_method : str, default="min"
        Method to aggregate expression values for complexes
    verbose : bool, default=False
        Print verbose output
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
        interaction_columns = ("A", "B")
    else:
        lr_pairs_renamed = lr_pairs
        interaction_columns = ("A", "B")

    # Generate expression values for protein complexes in PPI data
    if complex_sep is not None:
        if verbose:
            print("Getting expression values for protein complexes")
        _, _, _, _, complexes = get_genes_from_complexes(
            ppi_data=lr_pairs_renamed,
            complex_sep=complex_sep,
            interaction_columns=interaction_columns,
        )
        mod_rnaseq_matrices = [
            add_complexes_to_expression(
                rnaseq, complexes, agg_method=complex_agg_method
            )
            for rnaseq in pseudobulk_matrices_df
        ]
    else:
        mod_rnaseq_matrices = [df.copy() for df in pseudobulk_matrices_df]

    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score="expression_product",
        complex_sep=complex_sep,
        upper_letter_comparison=True,
        interaction_columns=interaction_columns,
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
    )

    # Only keep ppi_names to match the pairs used in the tensor in the interaction symbol column of the lr_pairs DataFrame
    lr_pair_names = lr_pairs["ligand"] + "^" + lr_pairs["receptor"]
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, sender, receiver)
    # To:   (context, sender, receiver, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def pseudobulk_cp_decomposition(
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
    # CP decomposition
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init="svd",  # Use SVD initialization
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)
    r2x = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return cp_weights, cp_factors, r2x


def save_ccc_rise_results(
    X: anndata.AnnData,
    cpd_factors: list[np.ndarray],
    weights: np.ndarray,
    lr_pairs: np.array,
):
    """Save CPD results in an AnnData object."""
    X.uns["A"] = cpd_factors[0]  # Condition factor
    X.uns["B"] = cpd_factors[1]  # Sender cell types factor
    X.uns["C"] = cpd_factors[2]  # Receiver cell types factor
    X.uns["D"] = cpd_factors[3]  # LR pairs factor
    X.uns["lr_pairs"] = lr_pairs["interaction_symbol"].values  # LR pairs
    X.uns["weights"] = weights  # Component weights

    return X


def get_genes_from_complexes(ppi_data, complex_sep="&", interaction_columns=("A", "B")):
    """
    Gets protein/gene names for individual proteins (subunits when in complex)
    in a list of PPIs. If protein is a complex, for example ProtA&ProtB, it will
    return ProtA and ProtB separately.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    col_a_genes : list
        List of protein/gene names for proteins and subunits in the first column
        of interacting partners.

    complex_a : list
        List of list of subunits of each complex that were present in the first
        column of interacting partners and that were returned as subunits in the
        previous list.

    col_b_genes : list
        List of protein/gene names for proteins and subunits in the second column
        of interacting partners.

    complex_b : list
        List of list of subunits of each complex that were present in the second
        column of interacting partners and that were returned as subunits in the
        previous list.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.
    """
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    col_a_genes = set()
    col_b_genes = set()

    complexes = dict()
    complex_a = set()
    complex_b = set()
    for idx, row in ppi_data.iterrows():
        prot_a = row[col_a]
        prot_b = row[col_b]

        if complex_sep in prot_a:
            comp = set([l for l in prot_a.split(complex_sep)])
            complexes[prot_a] = comp
            complex_a = complex_a.union(comp)
        else:
            col_a_genes.add(prot_a)

        if complex_sep in prot_b:
            comp = set([r for r in prot_b.split(complex_sep)])
            complexes[prot_b] = comp
            complex_b = complex_b.union(comp)
        else:
            col_b_genes.add(prot_b)

    return col_a_genes, complex_a, col_b_genes, complex_b, complexes


def add_complexes_to_expression(rnaseq_data, complexes, agg_method="min"):
    """
    Adds multimeric complexes into the gene expression matrix.
    Their gene expressions are the minimum expression value
    among the respective subunits composing them.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.

    agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    Returns
    -------
    tmp_rna : pandas.DataFrame
        Gene expression data for RNA-seq experiment containing multimeric
        complex names. Their gene expressions are the minimum expression value
        among the respective subunits composing them. Columns are
        cell-types/tissues/samples and rows are genes.
    """
    tmp_rna = rnaseq_data.copy()
    for k, v in complexes.items():
        if isinstance(v, set):
            v = list(v)
        elif isinstance(v, list):
            pass  # No need to convert, already a list
        else:
            raise ValueError(
                "Values in the `complexes`dictionary must be sets or lists."
            )
        if all(g in tmp_rna.index for g in v):
            df = tmp_rna.loc[v, :]
            if agg_method == "min":
                tmp_rna.loc[k] = df.min().values.tolist()
            elif agg_method == "mean":
                tmp_rna.loc[k] = df.mean().values.tolist()
            elif agg_method == "gmean":
                tmp_rna.loc[k] = df.apply(
                    lambda x: np.exp(np.mean(np.log(x)))
                ).values.tolist()
            else:
                ValueError("{} is not a valid agg_method".format(agg_method))
        else:
            tmp_rna.loc[k] = [0] * tmp_rna.shape[1]
    return tmp_rna
