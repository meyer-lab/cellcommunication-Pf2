import urllib.request
import os
import anndata
import pandas as pd
import numpy as np
import sparse


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
    else:
        print("File already exists. Loading data...")

    return anndata.read_h5ad(filename)


def import_ligand_receptor_pairs(filename="./data/Human-2020-Jin-LR-pairs.csv"):
    """Import ligand-receptor pairs from CellChat
    CellChat (Jin et al. 2021, Nature Communications"""

    url = "https://raw.githubusercontent.com/LewisLabUCSD/Ligand-Receptor-Pairs/refs/heads/master/Human/Human-2020-Jin-LR-pairs.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not os.path.exists(filename):
        print("Downloading data from GitHub...")
        urllib.request.urlretrieve(url, filename)
    else:
        print("File already exists. Loading data...")

    return pd.read_csv(filename)


def anndata_lrp_overlap(X: anndata.AnnData, df_lrp: pd.DataFrame):
    """Filter anndata to  include genes present in the ligand-receptor pairs data"""
    df_lrp = df_lrp[["ligand", "receptor"]]

    valid_mask = (df_lrp["ligand"].isin(X.var_names)) & (
        df_lrp["receptor"].isin(X.var_names)
    )
    df_lrp = df_lrp[valid_mask].copy().reset_index(drop=True)

    genes_to_keep = list(set(df_lrp["ligand"]) | set(df_lrp["receptor"]))

    return X[:, genes_to_keep], df_lrp


def add_cond_idxs(
    X: anndata.AnnData,
    condition_name: str,
) -> anndata.AnnData:
    """Add condition-specific indices to AnnData object"""
    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    return X


def anndata_to_tensor(X: anndata.AnnData) -> list:
    """
    Convert AnnData to a list of 3D sparse tensors for each condition.
    Each sample can have different numbers of sender/receiver cells.
    """
    samples = X.obs["condition_unique_idxs"].unique()
    lr_pairs = X.var_names
    tensor_list = []

    for sample in samples:
        # Filter data for this sample
        sample_data = X[X.obs["condition_unique_idxs"] == sample]

        # Get unique sender/receiver types for this sample
        sender_types = sample_data.obs["sender"].unique()
        receiver_types = sample_data.obs["receiver"].unique()

        # Create mappings specific to this sample
        sender_map = {ct: i for i, ct in enumerate(sender_types)}
        receiver_map = {ct: i for i, ct in enumerate(receiver_types)}

        # Get indices for all dimensions
        sender_indices = np.array([sender_map[s] for s in sample_data.obs["sender"]])
        receiver_indices = np.array(
            [receiver_map[r] for r in sample_data.obs["receiver"]]
        )

        # Create coordinates for 3D tensor
        sender_rep = np.repeat(sender_indices, len(lr_pairs))
        receiver_rep = np.repeat(receiver_indices, len(lr_pairs))
        lr_rep = np.tile(np.arange(len(lr_pairs)), len(sample_data))

        # Stack coordinates
        coords = np.stack([sender_rep, receiver_rep, lr_rep])
        shape = (len(sender_types), len(receiver_types), len(lr_pairs))
        values = sample_data.X.toarray().flatten()

        tensor = sparse.COO(coords, values, shape=shape)
        tensor_list.append(tensor)

    return tensor_list
