"""
Figure A3l_m: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
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
)
from .commonFuncs.plotFactors import plot_pair_lr_factors


def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))  # 1 row, 3 columns for 3 L-R  pairs
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
    # X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

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
    # X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    print("Epithelial sender cells:", X_mdc_sender.shape)
    print("Epithelial receiver cells:", X_mdc_receiver.shape)

    # Calculate amount of cells that fall into each label category and put in boxplot
    label_counts = X_mdc_sender.obs["Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    print("Sender cell label counts:\n", label_counts)

    sns.barplot(data=label_counts, x="Label", y="Count", ax=ax[0])
    ax[0].set_title("Sender Cell Label Counts")
    label_counts_receiver = X_mdc_receiver.obs["Label"].value_counts().reset_index()
    label_counts_receiver.columns = ["Label", "Count"]
    print("Receiver cell label counts:\n", label_counts_receiver)

    sns.barplot(data=label_counts_receiver, x="Label", y="Count", ax=ax[1])

    # Plot distribution of factor values for each label
    sns.boxplot(
        data=X_mdc_sender.obs,
        x="Label",
        y=X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp1 - 1],
        ax=ax[2],
    )
    ax[2].set_title(f"Sender Epithelial Cells {ccc_rise_cmp1} Distribution by Label")
    sns.boxplot(
        data=X_mdc_receiver.obs,
        x="Label",
        y=X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp1 - 1],
        ax=ax[3],
    )
    ax[3].set_title(f"Receiver Factor {ccc_rise_cmp1} Distribution by Label")

    sns.boxplot(
        data=X_mdc_sender.obs,
        x="Label",
        y=X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp2 - 1],
        ax=ax[4],
    )
    ax[4].set_title(f"Sender Factor {ccc_rise_cmp2} Distribution by Label")
    sns.boxplot(
        data=X_mdc_receiver.obs,
        x="Label",
        y=X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp2 - 1],
        ax=ax[5],
    )
    ax[5].set_title(f"Receiver Factor {ccc_rise_cmp2} Distribution by Label")

    plot_pair_lr_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[6])
    # Make axis symmetrical check both axes x and y
    xlim = np.max(np.abs(ax[6].get_xlim()))
    ylim = np.max([xlim, np.max(np.abs(ax[6].get_ylim()))])
    # Choose higher one
    lim = max(xlim, ylim)

    ax[6].set_xlim(-lim, lim)
    ax[6].set_ylim(-lim, lim)

    # Plot condition factors with condition label as well with A matrix
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    # Reorder index based on np.unique
    sample_to_group = sample_to_group.loc[
        np.unique(X.obs[condition_column], return_index=True)[0]
    ]
    pal = sns.color_palette(palette="Set2", n_colors=len(sample_to_group.unique()))
    pal = pal.as_hex()
    color_map = {k: v for k, v in zip(sample_to_group.unique(), pal)}

    # Map colors to each unique sample/condition based on their group
    colors = [
        color_map[sample_to_group.loc[condition]] for condition in sample_to_group.index
    ]

    # Compare component values for the two decompositions on scatter plot and pearson correlation
    # Log both axes to better visualize the spread.
    A_factor = X.uns["A"]

    ax[7].scatter(
        A_factor[:, ccc_rise_cmp1 - 1], A_factor[:, ccc_rise_cmp2 - 1], c=colors
    )
    ax[7].set_xscale("log")
    ax[7].set_yscale("log")
    r = np.corrcoef(A_factor[:, ccc_rise_cmp1 - 1], A_factor[:, ccc_rise_cmp2 - 1])[
        0, 1
    ]
    print("Condition factor correlation:", r)

    ax[7].set_xlabel(f"CCC-RISE Condition Component {ccc_rise_cmp1}")
    ax[7].set_ylabel(f"CCC-RISE Condition Component {ccc_rise_cmp2}")

    return f
