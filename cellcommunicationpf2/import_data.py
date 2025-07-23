import os
import urllib.request

import anndata
import pandas as pd

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
        Annotated data matrix.
    """
    url = "https://zenodo.org/record/7535867/files/BALF-COVID19-Liao_et_al-NatMed-2020.h5ad"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print("Downloading data from Zenodo...")
        urllib.request.urlretrieve(url, filename)

    return anndata.read_h5ad(filename)


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


def anndata_lrp_overlap(X: anndata.AnnData, df_lrp: pd.DataFrame):
    """Filter anndata to  include genes present in the ligand-receptor pairs data"""
    df_lrp = df_lrp[["ligand", "receptor"]]

    valid_mask = (df_lrp["ligand"].isin(X.var_names)) & (
        df_lrp["receptor"].isin(X.var_names)
    )
    df_lrp = df_lrp[valid_mask].copy().reset_index(drop=True)

    genes_to_keep = list(set(df_lrp["ligand"]) | set(df_lrp["receptor"]))

    return X[:, genes_to_keep], df_lrp


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
