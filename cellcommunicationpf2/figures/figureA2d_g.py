"""
Figure A2d-g: Comparison of CCC-RISE and Pseudobulk CPD on BALF COVID-19
"""

import numpy as np
import anndata
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_yaxis


def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    X_pseudo = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19_pseudobulk.h5ad")
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
    A_factor_pseudo = X_pseudo.uns["A"]

    cmp_X = 6
    cmp_X_pseudo = 8

    ax[0].scatter(
        A_factor[:, cmp_X - 1], A_factor_pseudo[:, cmp_X_pseudo - 1], c=colors
    )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    r = np.corrcoef(A_factor[:, cmp_X - 1], A_factor_pseudo[:, cmp_X_pseudo - 1])[0, 1]
    print("Condition factor correlation:", r)

    ax[0].set_xlabel(f"CCC-RISE Condition Component {cmp_X}")
    ax[0].set_ylabel(f"Pseudobulk CPD Condition Component {cmp_X_pseudo}")

    # Compare component values for the two decompositions on scatter plot and pearson correlation for the LR pairs.
    lr_factor = X.uns["D"]
    lr_factor_pseudo = X_pseudo.uns["D"]

    # Black points and blue center for points
    sns.scatterplot(
        x=lr_factor[:, cmp_X - 1],
        y=lr_factor_pseudo[:, cmp_X_pseudo - 1],
        ax=ax[1],
        color="black",
    )
    r = np.corrcoef(lr_factor[:, cmp_X - 1], lr_factor_pseudo[:, cmp_X_pseudo - 1])[
        0, 1
    ]
    print("LR factor correlation:", r)

    ax[1].set_xlabel(f"CCC-RISE LR Component {cmp_X}")
    ax[1].set_ylabel(f"Pseudobulk CPD LR Component {cmp_X_pseudo}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    # Make axes symmetric and have the same ticks
    max_val = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
    min_val = min(ax[1].get_xlim()[0], ax[1].get_ylim()[0])
    ax[1].set_xlim(min_val, max_val)
    ax[1].set_ylim(min_val, max_val)
    ticks = np.logspace(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)), num=5)
    ax[1].set_xticks(ticks)
    ax[1].set_yticks(ticks)

    # # Find which component values are most correlated between the two
    # # decompositions for both LR and conditions and show heatmap for both.
    # # Using divering color map to show negative correlations as well.
    cmp = 8
    r_mat = np.zeros((cmp, cmp))
    for i in range(cmp):
        for j in range(cmp):
            r_mat[i, j] = pearsonr(A_factor[:, i], A_factor_pseudo[:, j])[0]
    r_df = pd.DataFrame(
        r_mat,
        index=[f"{i + 1}" for i in range(cmp)],
        columns=[f"{i + 1}" for i in range(cmp)],
    )
    sns.heatmap(r_df, cmap="coolwarm", ax=ax[2], vmin=-1, vmax=1)
    ax[2].set_title("Condition Factor Correlation")
    rotate_yaxis(ax[2], rotation=0)
    ax[2].set_xlabel("Pseudobulk CPD Condition Component")
    ax[2].set_ylabel("CCC-RISE Condition Component")

    r_mat = np.zeros((cmp, cmp))
    for i in range(cmp):
        for j in range(cmp):
            r_mat[i, j] = pearsonr(lr_factor[:, i], lr_factor_pseudo[:, j])[0]
    r_df = pd.DataFrame(
        r_mat,
        index=[f"{i + 1}" for i in range(cmp)],
        columns=[f"{i + 1}" for i in range(cmp)],
    )
    sns.heatmap(r_df, cmap="coolwarm", ax=ax[3], vmin=-1, vmax=1)
    ax[3].set_title("LR Factor Correlation")
    rotate_yaxis(ax[3], rotation=0)
    ax[3].set_xlabel("Pseudobulk CPD LR Component")
    ax[3].set_ylabel("CCC-RISE LR Component")

    return f
