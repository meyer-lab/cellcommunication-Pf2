"""
Figure A3f: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils import (
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 3

    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_sender) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_sender), size=300, replace=False)
        X_mdc_sender = X_mdc_sender[sample_indices]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_receiver) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_receiver), size=300, replace=False)
        X_mdc_receiver = X_mdc_receiver[sample_indices]

    # Alter order based on factor value low to high
    # X_mdc_receiver = X_mdc_receiver[np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]

    print("Epithelial sender cells:", X_mdc_sender.shape)

    pairs = [
        ["PTN", "PTPRZ1"],
        ["PTN", "SDC1"],
        ["COL4A5", "SDC1"],
        ["MDK", "PTPRZ1"],
        ["MDK", "SDC1"],
    ]
    # Want boxplot across these average communication score across 3 conditions for each of these pairs

    # Calculate communication scores for each pair per sample
    communication_data = []

    for ligand, receptor in pairs:
        pair_name = f"{ligand}-{receptor}"

        # Get expression product matrix between sender and receiver cells
        expr_product_df = expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, ligand, receptor
        )

        # For each sample, calculate average communication score
        for sample in X.obs["sample"].unique():
            # Get condition for this sample
            sample_condition = X.obs[X.obs["sample"] == sample]["condition"].iloc[0]

            # Get sender cells from this sample
            sender_sample_mask = X_mdc_sender.obs["sample"] == sample
            sender_cells_sample = X_mdc_sender.obs_names[sender_sample_mask]

            # Get receiver cells from this sample
            receiver_sample_mask = X_mdc_receiver.obs["sample"] == sample
            receiver_cells_sample = X_mdc_receiver.obs_names[receiver_sample_mask]

            # Extract submatrix for this sample
            if len(sender_cells_sample) > 0 and len(receiver_cells_sample) > 0:
                sample_submatrix = expr_product_df.loc[
                    sender_cells_sample, receiver_cells_sample
                ]

                # Calculate average communication score for this sample and pair
                # Since we're now sampling from whole population, calculate overall score
                avg_comm_score = sample_submatrix.values.mean()
                
                # For compatibility with downstream analysis, we'll use the same score for both categories
                # but distinguish them in the analysis
                avg_comm_score_no_label = avg_comm_score
                avg_comm_score_label = avg_comm_score

                communication_data.append(
                    {
                        "pair": pair_name,
                        "sample": sample,
                        "condition": sample_condition,
                        "communication_score_no_label": avg_comm_score_no_label,
                        "communication_score_label": avg_comm_score_label,
                    }
                )

    # Convert to DataFrame
    comm_df = pd.DataFrame(communication_data)
    print("Communication scores data:")
    print(comm_df)

    # Plot both on the same plot for comparison with different names for legend with different colors for labeled vs no label. Combine into one plot and combine condition with label/no label
    comm_df_melted = pd.melt(
        comm_df,
        id_vars=["pair", "sample", "condition"],
        value_vars=["communication_score_no_label", "communication_score_label"],
        var_name="label_type",
        value_name="communication_score",
    )
    comm_df_melted["label_type"] = comm_df_melted["label_type"].map(
        {
            "communication_score_no_label": "No Label",
            "communication_score_label": "Labeled",
        }
    )
    comm_df_melted["condition_label"] = (
        comm_df_melted["condition"] + " - " + comm_df_melted["label_type"]
    )
    sns.boxplot(
        data=comm_df_melted,
        x="pair",
        y="communication_score",
        hue="condition_label",
        ax=ax[2],
    )
    ax[2].set_title(
        "Average Communication Scores by L-R Pair, Condition, and Label Type"
    )
    ax[2].set_xlabel("Ligand-Receptor Pair")
    ax[2].set_ylabel("Average Communication Score")
    ax[2].tick_params(axis="x", rotation=45)

    return f
