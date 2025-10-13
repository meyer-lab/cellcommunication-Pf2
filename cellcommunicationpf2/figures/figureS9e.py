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
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    sample_id = "dsco_id"
    condition_id = "alad_status"
    celltype_id = "broad_cell_type"
    
    # Group recovered and declined into ALAD category
    X.obs["alad_status"] = X.obs["ALADstatus"].astype(str).replace({"recovered": "ALAD", "declined": "ALAD"})
    
    ccc_rise_cmp = 12

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Macrophages"]
    X_mdc_sender = add_obs_cmp_label(
        X_mdc_sender, cmp=ccc_rise_cmp, pos=False, top_perc=1, type="sender"
    )
    X_mdc_sender = add_obs_cmp_unique_one(X_mdc_sender, cmp=ccc_rise_cmp)
    # X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Macrophages")]
    X_mdc_receiver = add_obs_cmp_label(
        X_mdc_receiver, cmp=ccc_rise_cmp, pos=False, top_perc=1, type="receiver"
    )
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    # X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    # X_mdc_receiver = X_mdc_receiver[np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]



    pairs = [
        ["GAS6", "MERTK"],
        ["SPP1", "ITGAV"],  # Use ITGAV as primary receptor component
        ["ANGPTL4", "ITGB3"],  # Use ITGB3 as primary receptor component
        ["SPP1", "ITGB3"],  # Use ITGB3 as primary receptor component
        ["ANGPTL4", "ITGAV"]  # Use ITGAV as primary receptor component
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
        for sample in X.obs[sample_id].unique():
            # Get condition for this sample
            sample_condition = X.obs[X.obs[sample_id] == sample][condition_id].iloc[0]

            # Get sender cells from this sample
            sender_sample_mask = X_mdc_sender.obs[sample_id] == sample
            sender_cells_sample = X_mdc_sender.obs_names[sender_sample_mask]

            # Get receiver cells from this sample
            receiver_sample_mask = X_mdc_receiver.obs[sample_id] == sample
            receiver_cells_sample = X_mdc_receiver.obs_names[receiver_sample_mask]

            # Extract submatrix for this sample
            if len(sender_cells_sample) > 0 and len(receiver_cells_sample) > 0:
                sample_submatrix = expr_product_df.loc[
                    sender_cells_sample, receiver_cells_sample
                ]

                # Calculate average communication score for this sample and pair for no label and labeled cells only
                avg_comm_score_no_label = sample_submatrix[
                    sample_submatrix.index.isin(
                        X_mdc_sender.obs_names[X_mdc_sender.obs["Label"] == "NoLabel"]
                    )
                ].values.mean()
                avg_comm_score_label = sample_submatrix[
                    sample_submatrix.index.isin(
                        X_mdc_sender.obs_names[X_mdc_sender.obs["Label"] != "NoLabel"]
                    )
                ].values.mean()

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
