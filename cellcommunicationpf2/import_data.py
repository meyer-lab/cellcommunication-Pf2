import os
import io
import zstandard as zstd
from functools import lru_cache
import urllib
import numpy as np
from scipy.sparse import issparse, csr_array
import pandas as pd
import anndata
import pandas as pd
# from parafac2.normalize import prepare_dataset


# The below code is taken directly from https://github.com/earmingol/cell2cell/blob/master/cell2cell/datasets/anndata.py
def import_balf_covid(filename="./data/BALF-COVID19-Liao_et_al-NatMed-2020.h5ad", gene_threshold: float = 0.01):
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

    assert hasattr(adata.X, "nnz"), "adata.X should be a sparse matrix"
    
    print(adata)

    return prepare_dataset(adata, condition_name="sample", geneThreshold=gene_threshold)


@lru_cache(maxsize=1)
def import_ligand_receptor_pairs(filename="./Human-2020-Jin-LR-pairs.csv.zst"):
    """Import ligand-receptor pairs from a zstd-compressed CSV with caching.

    The data is cached in memory after first load for improved performance.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Ligand-receptor pairs file not found: {filename}\n"
            "Please ensure the compressed file is present in the repository."
        )

    print("Loading ligand-receptor pairs data (zstd compressed)...")
    with open(filename, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            df = pd.read_csv(text_stream)
    
    # df['receptor'] = lr_pairs['receptor'].str.replace('_', '&')

    print(f"Cached {len(df)} ligand-receptor pairs")
    return df


def add_cond_idxs(X, condition_key):
    """Add unique condition indices to an AnnData object."""
    # Create a copy to avoid modifying a view
    X = X.copy()

    # Get unique conditions and map to indices
    conditions = X.obs[condition_key].unique()
    conditions = ["C51", "C52", "C100", "C141", "C142", "C143", "C144", "C145", "C146","C148", "C149", "C152"]
    condition_map = {cond: i for i, cond in enumerate(conditions)}
    print(f"Mapping conditions to indices: {condition_map}")
    # Add indices to obs
    X.obs["condition_unique_idxs"] = X.obs[condition_key].map(condition_map).values

    return X



def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float, deviance=False
) -> anndata.AnnData:
    assert issparse(X.X)
    X.X = csr_array(X.X)
    assert np.amin(X.X.data) >= 0.0

    # Filter out genes with too few reads, and cells with fewer than 10 counts
    # X = X[X.X.sum(axis=1) > 10, X.X.mean(axis=0) > geneThreshold]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())
    X.X = csr_array(X.X)

    # Convert counts to floats
    if issubclass(X.X.dtype.type, int | np.integer):
        X.X.data = X.X.data.astype(np.float32)

    # if deviance:
    #     X.X = get_deviance(X.X)
    # else:
    #     ## Normalize total counts per cell
    #     # Keep the counts on a reasonable scale to avoid accuracy issues
    #     counts_per_cell = X.X.sum(axis=1)
    #     counts_per_cell /= np.median(counts_per_cell)
    #     # inplace csr row scale
    #     X.X.data /= np.repeat(counts_per_cell, np.diff(X.X.indptr))

    #     # Scale genes by sum, inplace csr col scale
    #     X.X.data /= X.X.sum(axis=0).take(X.X.indices, mode="clip")

    #     # Transform values
    #     X.X.data = np.log10((1000.0 * X.X.data) + 1.0)

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    X.var["means"] = X.X.mean(axis=0)

    return X