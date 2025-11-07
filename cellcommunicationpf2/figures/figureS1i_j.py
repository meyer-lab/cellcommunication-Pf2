"""
Figure S2i_j: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 6

    X_mdc_sender = X[X.obs["celltype"] == "Macrophages"]
    print("Macrophage sender cells:", X_mdc_sender.shape)

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["celltype"] == "Macrophages")]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, "CCL19", "CCR7")
    df = group_matrix(df)
    sns.heatmap(df, ax=ax[0], cmap="viridis", vmax=0.12)

    X_b_receiver = X[(X.obs["celltype"] == "NK")]
    print("NK cell receiver shape:", X_b_receiver.shape)

    # Alter order based on factor value low to high
    X_b_receiver = X_b_receiver[
        np.argsort(X_b_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    df = expression_product_matrix(X_mdc_sender, X_b_receiver, "CCL19", "CCR7")
    df = group_matrix(df)
    sns.heatmap(df, ax=ax[1], cmap="viridis", vmax=0.12)

    ax[0].set_xlabel("Receiver Cells (Macrophages)")
    ax[0].set_ylabel("Sender Cells (Macrophages)")
    ax[1].set_xlabel("Receiver Cells (NK cells)")
    ax[1].set_ylabel("Sender Cells (Macrophages)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_title("CCL19-CCR7 Interaction")
    ax[1].set_title("CCL19-CCR7 Interaction")

    return f


def group_matrix(df):
    """
    Groups a DataFrame into a 10x10 matrix by binning rows and columns and averaging within bins.
    Prints shape information for debugging.
    """
    print(f"Original matrix shape: {df.shape}")
    n_rows = len(df)
    n_cols = len(df.columns)
    row_group_size = n_rows // 10
    col_group_size = n_cols // 10
    print(f"Row group size: {row_group_size}, Col group size: {col_group_size}")
    row_groups = np.arange(n_rows) // row_group_size
    col_groups = np.arange(n_cols) // col_group_size
    row_groups = np.clip(row_groups, 0, 9)
    col_groups = np.clip(col_groups, 0, 9)
    df_grouped = df.groupby(row_groups).mean()
    df_grouped = df_grouped.groupby(col_groups, axis=1).mean()
    print(f"Final grouped matrix shape: {df_grouped.shape}")
    print(df_grouped)
    return df_grouped
