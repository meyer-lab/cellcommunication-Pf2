import anndata
import numpy as np
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from .cc_pf2 import cc_pf2, standardize_cc_pf2
from .import_data import add_cond_idxs
from sklearn.linear_model import LinearRegression
import pandas as pd


def resample(data: anndata.AnnData, random_seed: int = None) -> anndata.AnnData:
    """Perform stratified bootstrap sampling by resampling cells within each sample.

    This maintains the same number of cells per sample in the resampled dataset.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    resampled_indices = []
    for sample in data.obs["sample"].unique():
        # Get indices of cells in this sample
        sample_cell_indices = np.where(data.obs["sample"] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Sample with replacement within this sample's cells
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData from resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


def calculate_fms(A: anndata.AnnData, B: anndata.AnnData) -> float:
    """Calculate FMS between two CC-PF2 decompositions stored in AnnData objects.

    Skips comparison of sender/receiver factors (modes 1 and 2) as they are most variable.
    """
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))

    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]]
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    return fms(A_CP, B_CP, consider_weights=False, skip_mode=[1, 2])


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]

    counts = np.zeros((np.amax(sgIndex.to_numpy()) + 1, 1))

    cond_mean = np.linalg.norm(X.uns["Pf2_A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct


def run_cc_pf2_workflow(
    adata: anndata.AnnData,
    rise_rank: int,
    lr_pairs: pd.DataFrame,
    cp_rank: int | None = None,
    condition_column: str = "sample",
    n_iter_max: int = 100,
    tol: float = 1e-3,
    random_state: int | None = None,
) -> tuple[anndata.AnnData, float]:
    """
    Executes the complete CC-PF2 workflow: decomposition, standardization,
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

    # 1. Run the CC-PF2 decomposition
    results, r2x, filtered_lr_pairs = cc_pf2(
        adata,
        rise_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        cp_rank=cp_rank,
        random_state=random_state,
    )
    (cp_weights, factors), projections = results

    # 2. Standardize the factors for interpretability
    weights, factors = standardize_cc_pf2(cp_weights, factors)

    # Store factors in AnnData object for easy access by plotting functions
    adata.uns["Pf2_A"] = factors[0]  # Condition factor
    adata.uns["Pf2_B"] = factors[1]  # Sender cells factor
    adata.uns["Pf2_C"] = factors[2]  # Receiver cells factor
    adata.uns["Pf2_D"] = factors[3]  # LR pairs factor

    # Store the LR pairs and R2X
    adata.uns["Pf2_lr_pairs"] = filtered_lr_pairs
    adata.uns["Pf2_r2x"] = r2x
    adata.uns["Pf2_weights"] = weights
    adata.uns["Pf2_projections"] = projections

    return adata, r2x


def pseudobulk_X(X: anndata, condition_name: str, groupby: str, type: str) -> list[pd.DataFrame]:
    """
    Calculate average gene expression for each groupby in each sample
    """
    # Get unique samples and cell types
    samples = ["C51", "C52", "C100", "C141", "C142", "C143", "C144", "C145", "C146","C148", "C149", "C152"]
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
            if adata_subset.n_obs > 0 and type is "mean":
                mean_expression = np.mean(adata_subset.X.toarray(), axis=0)
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = mean_expression[i]
            elif adata_subset.n_obs > 0 and type is "fraction":
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


def pseudobulk_X_save(X: anndata, condition_name: str, groupby: str, type: str) -> list[pd.DataFrame]:
    """
    Calculate average gene expression for each groupby in each sample
    """
    # Get unique samples and cell types
    samples = ["C51", "C52", "C100", "C141", "C142", "C143", "C144", "C145", "C146","C148", "C149", "C152"]
    groupby_names = X.obs[groupby].unique()
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
            if adata_subset.n_obs > 0 and type is "mean":
                mean_expression = np.mean(adata_subset.X.toarray(), axis=0)
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = mean_expression[i]
            elif adata_subset.n_obs > 0 and type is "fraction":
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
        results_df = results_df.T
        results_df.to_csv(f"{sample}_pseudobulk.csv")