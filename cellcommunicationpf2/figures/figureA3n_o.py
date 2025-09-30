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
    X_mdc_sender = add_obs_cmp_both_label(
        X_mdc_sender,
        cmp1=ccc_rise_cmp1,
        cmp2=ccc_rise_cmp2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="sender",
    )
    X_mdc_sender = add_obs_cmp_unique_two(
        X_mdc_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2
    )
    X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_mdc_receiver = add_obs_cmp_both_label(
        X_mdc_receiver,
        cmp1=ccc_rise_cmp1,
        cmp2=ccc_rise_cmp2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="receiver",
    )
    X_mdc_receiver = add_obs_cmp_unique_two(
        X_mdc_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2
    )
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

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

        # Get sender and receiver labels
        sender_labels = X_mdc_sender.obs["Label"].values
        receiver_labels = X_mdc_receiver.obs["Label"].values

        # Calculate average communication score for each sender-receiver label combination
        for sender_label in np.unique(sender_labels):
            for receiver_label in np.unique(receiver_labels):
                # Get indices for this label combination
                sender_idx = np.where(sender_labels == sender_label)[0]
                receiver_idx = np.where(receiver_labels == receiver_label)[0]

                if len(sender_idx) > 0 and len(receiver_idx) > 0:
                    # Extract submatrix for this label combination
                    submatrix = df.iloc[sender_idx, receiver_idx]

                    # Calculate average communication score
                    avg_comm_score = submatrix.values.mean()

                    communication_data.append(
                        {
                            "pair": pair_name,
                            "sender_label": sender_label,
                            "receiver_label": receiver_label,
                            "label_combination": f"{sender_label}→{receiver_label}",
                            "communication_score": avg_comm_score,
                            "n_sender_cells": len(sender_idx),
                            "n_receiver_cells": len(receiver_idx),
                            "n_interactions": len(sender_idx) * len(receiver_idx),
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
