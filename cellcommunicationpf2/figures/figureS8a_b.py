"""
Figure S8a-b: Heatmaps of ligand-receptor interaction scores between sender and receiver cells for specific cell types and components in CCC-RISE on BALF alad data.
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

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    ccc_rise_cmp = 12

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Macrophages"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Macrophages")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(-X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]


    pairs = [["GAS6", "MERTK"]]
    for i, (lig, rec) in enumerate(pairs):
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        print(f"Original matrix shape: {df.shape}")
        
        # Group rows and columns into exactly 10 brackets each to create a 10x10 matrix
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Calculate group sizes to get exactly 10 groups
        row_group_size = n_rows // 10
        col_group_size = n_cols // 10
        
        print(f"Row group size: {row_group_size}, Col group size: {col_group_size}")
        
        # Create grouping arrays for exactly 10 groups
        row_groups = np.arange(n_rows) // row_group_size
        col_groups = np.arange(n_cols) // col_group_size
        
        # Ensure we don't exceed 10 groups (clip any remainder cells to group 9)
        row_groups = np.clip(row_groups, 0, 9)
        col_groups = np.clip(col_groups, 0, 9)
        
        # Group and take averages
        df_grouped = df.groupby(row_groups).mean()
        df_grouped = df_grouped.groupby(col_groups, axis=1).mean()
        
        print(f"Final grouped matrix shape: {df_grouped.shape}")
        print(df_grouped)
        # Keep max value consistent across heatmaps for better comparison
        sns.heatmap(df_grouped, ax=ax[i], cmap="rocket")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        
    vmax = df_grouped.values.max()
        
    ccc_rise_cmp = 12

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Dendritic Cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(-X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]


    pairs = [["GAS6", "MERTK"]]
    for i, (lig, rec) in enumerate(pairs):
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        print(f"Original matrix shape: {df.shape}")
        
        # Group rows and columns into exactly 10 brackets each to create a 10x10 matrix
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Calculate group sizes to get exactly 10 groups
        row_group_size = n_rows // 10
        col_group_size = n_cols // 10
        
        print(f"Row group size: {row_group_size}, Col group size: {col_group_size}")
        
        # Create grouping arrays for exactly 10 groups
        row_groups = np.arange(n_rows) // row_group_size
        col_groups = np.arange(n_cols) // col_group_size
        
        # Ensure we don't exceed 10 groups (clip any remainder cells to group 9)
        row_groups = np.clip(row_groups, 0, 9)
        col_groups = np.clip(col_groups, 0, 9)
        
        # Group and take averages
        df_grouped = df.groupby(row_groups).mean()
        df_grouped = df_grouped.groupby(col_groups, axis=1).mean()
        
        print(f"Final grouped matrix shape: {df_grouped.shape}")
        print(df_grouped)
        # Keep max value consistent across heatmaps for better comparison
        sns.heatmap(df_grouped, ax=ax[i+2], cmap="rocket", vmax=vmax)
        ax[i+2].set_title(f"{lig}-{rec} Interaction")
        
    



    return f