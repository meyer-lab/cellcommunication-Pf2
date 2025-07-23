import os
import urllib.request

import anndata
import pandas as pd
import numpy as np

# Module-level cache for ligand-receptor pairs
_lr_pairs_cache = None


# The below code is taken directly from https://github.com/earmingol/cell2cell/blob/master/cell2cell/datasets/anndata.py
def import_balf_covid(filename="./data/BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"):
    """BALF samples from COVID-19 patients
    The data consists in 63k immune and epithelial cells in lungs
    from 3 control, 3 moderate COVID-19, and 6 severe COVID-19 patients.

    This dataset was previously published in [1], and this objects contains
    the raw counts for the annotated cell types available in:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926

    References:
    [1] Liao, M., Liu, Y., Yuan, J. et al.
        Single-cell landscape of bronchoalveolar immune cells in patients
        with COVID-19. Nat Med 26, 842–844 (2020).
        https://doi.org/10.1038/s41591-020-0901-9

    Parameters
    ----------
        filename : str, default='BALF-COVID19-Liao_et_al-NatMed-2020.h5ad'
            Path to the h5ad file in case it was manually downloaded.

    Returns
    -------
        Annotated data matrix with sparse X matrix preserved.
    """
    url = "https://zenodo.org/record/7535867/files/BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print("Downloading data from Zenodo...")
        urllib.request.urlretrieve(url, filename)

    print("Loading BALF COVID data (preserving sparse format)...")
    adata = anndata.read_h5ad(filename)
    
    # Ensure X matrix stays sparse
    if not hasattr(adata.X, 'nnz'):  # Check if it's not already sparse
        print("Warning: Data was loaded as dense, converting to sparse...")
        import scipy.sparse as sp
        adata.X = sp.csr_matrix(adata.X)
        # Calculate sparsity for dense matrix
        zeros = (adata.X == 0).sum()
        sparsity = (zeros / adata.X.size) * 100
        print(f"Converted to sparse: {sparsity:.2f}% zeros (sparse)")
    else:
        # Calculate correct sparsity percentage
        total_elements = adata.n_obs * adata.n_vars
        non_zero_pct = (adata.X.nnz / total_elements) * 100
        zero_pct = 100 - non_zero_pct
        
        print(f"Data loaded as sparse matrix: {type(adata.X).__name__}")
        print(f"  Non-zero elements: {adata.X.nnz:,} ({non_zero_pct:.2f}%)")
        print(f"  Zero elements: {total_elements - adata.X.nnz:,} ({zero_pct:.2f}%)")
        print(f"  Sparsity: {zero_pct:.2f}%")
    
    adata.obs_names_make_unique()  # Ensure unique cell names
    return filter_genes_by_threshold(adata)


def import_ligand_receptor_pairs(filename="./data/Human-2020-Jin-LR-pairs.csv"):
    """Import ligand-receptor pairs from CellChat with caching
    CellChat (Jin et al. 2021, Nature Communications)

    The data is cached in memory after first load for improved performance.
    """
    global _lr_pairs_cache

    # Return cached version if available
    if _lr_pairs_cache is not None:
        print("Using cached ligand-receptor pairs data")
        return _lr_pairs_cache.copy()

    url = "https://raw.githubusercontent.com/LewisLabUCSD/Ligand-Receptor-Pairs/refs/heads/master/Human/Human-2020-Jin-LR-pairs.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not os.path.exists(filename):
        print("Downloading ligand-receptor pairs from GitHub...")
        urllib.request.urlretrieve(url, filename)

    print("Loading ligand-receptor pairs data...")
    df = pd.read_csv(filename)

    # Cache the loaded data
    _lr_pairs_cache = df
    print(f"Cached {len(df)} ligand-receptor pairs")

    return df


def add_cond_idxs(X, condition_key):
    """Add unique condition indices to an AnnData object."""
    # Create a copy to avoid modifying a view
    X = X.copy()

    # Get unique conditions and map to indices
    conditions = X.obs[condition_key].unique()
    condition_map = {cond: i for i, cond in enumerate(conditions)}

    # Add indices to obs
    X.obs["condition_unique_idxs"] = X.obs[condition_key].map(condition_map).values

    return X

def filter_genes_by_threshold(adata, threshold=0):
    """
    Remove genes where all values are <= threshold.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object
    threshold : float, default=0
        Remove genes where ALL values are <= this threshold
        
    Returns
    -------
    anndata.AnnData
        Filtered AnnData object with genes removed
    """
    
    # Get max absolute value per gene and convert to a 1D numpy array
    gene_max = np.abs(adata.X).max(axis=0).toarray().flatten()

    genes_to_keep = gene_max > threshold
    
    # Total kept genes
    print(f"Keeping {np.sum(genes_to_keep)} genes with max value > {threshold}")
    
    return adata[:, genes_to_keep]
