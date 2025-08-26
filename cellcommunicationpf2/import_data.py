import os
import io
import zstandard as zstd
from functools import lru_cache
import urllib

import anndata
import pandas as pd
from parafac2.normalize import prepare_dataset


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
