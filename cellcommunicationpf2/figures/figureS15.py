"""
Figure A2k-l: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
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

    # Check unique sex values in the data
    print("Unique sex values:", X.obs["sex"].unique())
    print("Sex value counts:", X.obs["sex"].value_counts())

    pairs = [["GAS6", "MERTK"]]
    
    # Create a list to store vmax values for consistent scaling
    vmax_values = []
    
    # First pass: calculate all heatmaps to determine global vmax
    heatmap_data = {}
    axs = 0

    for j, sex in enumerate(["male", "female"]):
        # Filter data by sex
        X_sex = X[X.obs["sex"] == sex]
        print(f"\nProcessing {sex} patients: {len(X_sex)} cells")
        
        X_mdc_sender = X_sex[X_sex.obs["broad_cell_type"] == "Macrophages"]
        X_mdc_sender = X_mdc_sender[
            np.argsort(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
        ]

        X_mdc_receiver = X_sex[X_sex.obs["broad_cell_type"] == "Macrophages"]
        X_mdc_receiver = X_mdc_receiver[
            np.argsort(-X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
        ]
        
        print(f"{sex} - Sender cells: {len(X_mdc_sender)}, Receiver cells: {len(X_mdc_receiver)}")
        
        for i, (lig, rec) in enumerate(pairs):
            if len(X_mdc_sender) > 0 and len(X_mdc_receiver) > 0:
                df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
                print(f"{sex} - Original matrix shape: {df.shape}")
                
                # Group rows and columns into exactly 10 brackets each to create a 10x10 matrix
                n_rows = len(df)
                n_cols = len(df.columns)
                
                if n_rows > 0 and n_cols > 0:
                    # Calculate group sizes to get exactly 10 groups
                    row_group_size = max(1, n_rows // 10)
                    col_group_size = max(1, n_cols // 10)
                    
                    # Create grouping arrays for exactly 10 groups
                    row_groups = np.arange(n_rows) // row_group_size
                    col_groups = np.arange(n_cols) // col_group_size
                    
                    # Ensure we don't exceed 10 groups (clip any remainder cells to group 9)
                    row_groups = np.clip(row_groups, 0, 9)
                    col_groups = np.clip(col_groups, 0, 9)
                    
                    # Group and take averages
                    df_grouped = df.groupby(row_groups).mean()
                    df_grouped = df_grouped.groupby(col_groups, axis=1).mean()
                    
                    sns.heatmap(df_grouped, ax=ax[axs], cmap="rocket", vmax=.002)
                    ax[axs].set_title(f"{lig}-{rec} Interaction: {sex}")

                    axs += 1
                
  




    return f