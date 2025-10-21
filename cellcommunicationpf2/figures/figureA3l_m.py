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

    # Plot histograms of factor values instead of label counts
    import pandas as pd
    
    # Create histograms of factor values for sender cells
    ax[0].hist(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp1 - 1], bins=30, alpha=0.7, label=f'Component {ccc_rise_cmp1}')
    ax[0].hist(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp2 - 1], bins=30, alpha=0.7, label=f'Component {ccc_rise_cmp2}')
    ax[0].set_title("Sender Cell Factor Value Distribution")
    ax[0].set_xlabel("Factor Value")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()

    # Create histograms of factor values for receiver cells
    ax[1].hist(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp1 - 1], bins=30, alpha=0.7, label=f'Component {ccc_rise_cmp1}')
    ax[1].hist(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp2 - 1], bins=30, alpha=0.7, label=f'Component {ccc_rise_cmp2}')
    ax[1].set_title("Receiver Cell Factor Value Distribution") 
    ax[1].set_xlabel("Factor Value")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    # Plot scatter plots of factor values instead of boxplots by label
    ax[2].scatter(range(len(X_mdc_sender)), X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp1 - 1], alpha=0.6)
    ax[2].set_title(f"Sender Component {ccc_rise_cmp1} Factor Values")
    ax[2].set_xlabel("Cell Index")
    ax[2].set_ylabel("Factor Value")
    
    ax[3].scatter(range(len(X_mdc_receiver)), X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp1 - 1], alpha=0.6)
    ax[3].set_title(f"Receiver Component {ccc_rise_cmp1} Factor Values")
    ax[3].set_xlabel("Cell Index")
    ax[3].set_ylabel("Factor Value")

    ax[4].scatter(range(len(X_mdc_sender)), X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp2 - 1], alpha=0.6)
    ax[4].set_title(f"Sender Component {ccc_rise_cmp2} Factor Values")
    ax[4].set_xlabel("Cell Index") 
    ax[4].set_ylabel("Factor Value")
    
    ax[5].scatter(range(len(X_mdc_receiver)), X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp2 - 1], alpha=0.6)
    ax[5].set_title(f"Receiver Component {ccc_rise_cmp2} Factor Values")
    ax[5].set_xlabel("Cell Index")
    ax[5].set_ylabel("Factor Value")

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
