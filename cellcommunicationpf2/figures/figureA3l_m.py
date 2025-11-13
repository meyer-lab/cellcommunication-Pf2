"""
Figure A3l_m: Scatter plots of condition factors and ligand-receptor factors for CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from .commonFuncs.plotFactors import plot_pair_lr_factors


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2)) 
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp1 = 3
    ccc_rise_cmp2 = 5

    plot_pair_lr_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[0])

    xlim = np.max(np.abs(ax[0].get_xlim()))
    ylim = np.max([xlim, np.max(np.abs(ax[0].get_ylim()))])
    lim = max(xlim, ylim)
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)


    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    sample_to_group = sample_to_group.loc[
        np.unique(X.obs[condition_column], return_index=True)[0]
    ]
    pal = sns.color_palette(palette="Set2", n_colors=len(sample_to_group.unique()))
    pal = pal.as_hex()
    color_map = {k: v for k, v in zip(sample_to_group.unique(), pal)}
    colors = [
        color_map[sample_to_group.loc[condition]] for condition in sample_to_group.index
    ]

    A_factor = X.uns["A"]

    ax[2].scatter(
        A_factor[:, ccc_rise_cmp1 - 1], A_factor[:, ccc_rise_cmp2 - 1], c=colors
    )
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    r = np.corrcoef(A_factor[:, ccc_rise_cmp1 - 1], A_factor[:, ccc_rise_cmp2 - 1])[
            0, 1
        ]
    print("Condition factor correlation:", r)
    ax[2].set_xlabel(f"CCC-RISE Condition Component {ccc_rise_cmp1}")
    ax[2].set_ylabel(f"CCC-RISE Condition Component {ccc_rise_cmp2}")

    return f
