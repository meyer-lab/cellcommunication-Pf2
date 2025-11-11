"""
Figure A4g: Heatmaps of ligand-receptor expression products for CCL17-CCR4 and CCL22-CCR4
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    ccc_rise_cmp = 5

    X_mdc_sender = X[(X.obs["broad_cell_type"] == "CD4 T cells")]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]


    pairs = ["CD40LG", "CD40"]
    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1])
    df_grouped = group_matrix(df)
    # Keep max value consistent across heatmaps for better comparison
    sns.heatmap(df_grouped, ax=ax[0], cmap="rocket", vmax=0.003)
    ax[0].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[0].set_xlabel("Receiver Dendritic Cells")
    ax[0].set_ylabel("Sender CD4 T Cells")
    
    pairs = ["LTB", "LTBR"]
    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1], )
    df_grouped = group_matrix(df)
    # Keep max value consistent across heatmaps for better comparison
    sns.heatmap(df_grouped, ax=ax[1], cmap="rocket", vmax=0.003)
    ax[1].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[1].set_xlabel("Receiver Dendritic Cells")
    ax[1].set_ylabel("Sender CD4 T Cells")
    
    
    X_mdc_sender = X[(X.obs["broad_cell_type"] == "NK cells")]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]


    pairs = ["CD40LG", "CD40"]
    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1])
    df_grouped = group_matrix(df)
    # Keep max value consistent across heatmaps for better comparison
    sns.heatmap(df_grouped, ax=ax[2], cmap="rocket", vmax=0.003)
    ax[2].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[2].set_xlabel("Receiver Dendritic Cells")
    ax[2].set_ylabel("Sender NK Cells")
    

    
    
    
    
 
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