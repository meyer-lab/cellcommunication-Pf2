import anndata
import numpy as np
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from .ccc_rise import ccc_rise, standardize_cc_pf2
from .import_data import add_cond_idxs
from sklearn.linear_model import LinearRegression
import pandas as pd
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2, anndata_to_list
from tensorly.cp_tensor import CPTensor
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from .ccc_rise import calc_communication_score


def resample(data: anndata.AnnData, condition_name: str, random_seed: int = None) -> anndata.AnnData:
    """Perform stratified bootstrap sampling by resampling cells within each sample.

    This maintains the same number of cells per sample in the resampled dataset.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    resampled_indices = []
    for sample in data.obs[condition_name].unique():
        # Get indices of cells in this sample
        sample_cell_indices = np.where(data.obs[condition_name] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Sample with replacement within this sample's cells
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData from resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


def resample_tensor(interaction_tensors):
    """Bootstrap tensor by resampling last dimension"""
    indices = np.random.randint(0, interaction_tensors.shape[-1], size=interaction_tensors.shape[-1])
    return interaction_tensors[..., indices]

def rise_store_r2x(X: anndata.AnnData, rank: int, n_iter_max: int, tolerance: float, random_state: int = None):
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


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]

    counts = np.zeros((np.amax(sgIndex.to_numpy()) + 1, 1))

    cond_mean = np.linalg.norm(X.uns["A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["A"] / counts_correct


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
    doEmbedding: bool = True
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

    Returns
    -------
    tuple[anndata.AnnData, float]
        A tuple containing the updated AnnData object with stored results
        and the R2X value of the decomposition.
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


def pseudobulk_X(X: anndata, condition_name: str, groupby: str, type: str) -> list[pd.DataFrame]:
    """
    Calculate average gene expression for each groupby in each sample
    """
    # Get unique samples and cell types
    samples = np.unique(X.obs[condition_name].unique())
    groupby_names = np.unique(X.obs[groupby].unique())
    gene_names = X.var_names

    total_df = []
    for sample in samples:
        # Subset to current sample
        sample_mask = X.obs[condition_name] == sample
        X_sample = X[sample_mask, :]
        results = []
        for group_name in groupby_names:
            # Subset to current groupby
            group_mask = X_sample.obs[groupby] == group_name
            adata_subset = X_sample[group_mask, :]
            result_dict = {}
            if adata_subset.n_obs > 0 and type == "mean":
                mean_expression = np.mean(adata_subset.X.toarray(), axis=0)
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = mean_expression[i]
            elif adata_subset.n_obs > 0 and type == "fraction":
                ct_df = adata_subset.X.toarray()
                cell_fraction = ((ct_df > 0).sum(axis=0) / ct_df.shape[0])
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = cell_fraction[i]
            else:
                # Set zero expression for missing combinations
                for gene in gene_names:
                    result_dict[gene] = 0.0

            results.append(result_dict)

        # Dataframe with genes as rows and groupby as columns
        results_df = pd.DataFrame(results, columns=gene_names, index=groupby_names)
        total_df.append(results_df.T)

    return total_df



def calculate_interaction_tensor(X_filtered: anndata.AnnData, lr_pairs: pd.DataFrame, rise_rank: int):
    """Calculate interaction tensor from AnnData object using PARAFAC2 and communication scores."""
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
    
    return interaction_tensor


def run_fms_r2x_analysis(interaction_tensor: np.ndarray, rank_list: list[int] = None, runs: int = 1) -> pd.DataFrame:
    """Run FMS and R2X analysis across different CP ranks and bootstrap runs."""
    if rank_list is None:
        rank_list = list(range(1, 4, 2))
    
    fms_list = []
    r2xLists = []
    
    for i in range(runs):
        scores = []
        r2x_scores = []
        for j in rank_list:
            print(f"Run {i+1}, Rank {j}")
            boot_tensor = resample_tensor(interaction_tensor)
            cp_weights, cp_factors = parafac(
                tensor=interaction_tensor,
                rank=j,
                n_iter_max=1000,
                init="svd",  # Use SVD initialization
                normalize_factors=True,
            )
            r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
            cp_boot_weights, cp_boot_factors = parafac(
                tensor=boot_tensor,
                rank=j,
                n_iter_max=1000,
                init="svd",  # Use SVD initialization
                normalize_factors=True,
            )
            fms_score = calculate_fms_cpd(cp_weights, cp_factors, cp_boot_weights, cp_boot_factors)
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
        {"Run": runsList_df, "Component": ranksList_df, "FMS": fmsList_df, "R2X": r2xList_df}
    )
    
    return df