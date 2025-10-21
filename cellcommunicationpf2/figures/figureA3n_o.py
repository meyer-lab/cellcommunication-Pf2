"""
Figure A3n-o: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    add_obs_cmp_both_label,
    add_obs_cmp_unique_two,
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((10, 10), (3, 3))  # 1 row, 3 columns for 3 L-R  pairs
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp1 = 3
    ccc_rise_cmp2 = 5

    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_sender) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_sender), size=300, replace=False)
        X_mdc_sender = X_mdc_sender[sample_indices]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_receiver) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_receiver), size=300, replace=False)
        X_mdc_receiver = X_mdc_receiver[sample_indices]

    print("Epithelial sender cells:", X_mdc_sender.shape)
    print("Epithelial receiver cells:", X_mdc_receiver.shape)

    # Calculate average communication scores for each label category
    import pandas as pd

    # Define ligand-receptor pairs to analyze
    pairs = [
        ["PTN", "PTPRZ1"],
        ["PTN", "SDC1"],
        ["COL4A5", "SDC1"],
        ["CDH1", "CDH1"],
        ["OCLN", "OCLN"],
        ["PRSS3", "F2RL1"],
    ]

    # Collect communication scores for all label combinations
    communication_data = []

    for lig, rec in pairs:
        pair_name = f"{lig}-{rec}"

        # Get expression product matrix
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)

        # Since we're sampling from whole population, calculate overall communication score
        avg_comm_score = df.values.mean()

        communication_data.append(
            {
                "pair": pair_name,
                "sender_label": "All_Cells",
                "receiver_label": "All_Cells", 
                "label_combination": "All_Cells→All_Cells",
                "communication_score": avg_comm_score,
                "n_sender_cells": len(X_mdc_sender),
                "n_receiver_cells": len(X_mdc_receiver),
                "n_interactions": len(X_mdc_sender) * len(X_mdc_receiver),
            }
        )

    # Convert to DataFrame
    comm_df = pd.DataFrame(communication_data)

    # Create separate heatmap for each ligand-receptor pair
    for i, pair_name in enumerate(pairs):
        pair_label = f"{pair_name[0]}-{pair_name[1]}"

        # Filter data for this specific pair
        pair_data = comm_df[comm_df["pair"] == pair_label]

        if len(pair_data) == 0:
            print(f"No data found for pair: {pair_label}")
            continue

        # Create pivot table for this pair
        pivot_data = pair_data.pivot_table(
            values="communication_score",
            index="sender_label",
            columns="receiver_label",
            aggfunc="mean",
        )

        print(f"\nPivot data for {pair_label}:")
        print(pivot_data)
        # Normalize from 0 to 1 for better visualization
        pivot_data = (pivot_data - pivot_data.min().min()) / (
            pivot_data.max().max() - pivot_data.min().min()
        )

        # Create heatmap
        sns.heatmap(
            pivot_data,
            # annot=True,
            fmt=".4f",
            cmap="Purples",
            cbar_kws={"label": "Avg Communication Score"},
            ax=ax[i],
        )

        ax[i].set_title(f"{pair_label} Communication Scores\n(Sender → Receiver)")
        ax[i].set_xlabel("Receiver Label")
        ax[i].set_ylabel("Sender Label")

    return f
