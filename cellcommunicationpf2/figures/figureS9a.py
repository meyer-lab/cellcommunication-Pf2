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

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    ccc_rise_cmp1 = 12
    ccc_rise_cmp2 = 14


    plot_pair_lr_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[6])
    # Make axis symmetrical check both axes x and y
    xlim = np.max(np.abs(ax[6].get_xlim()))
    ylim = np.max([xlim, np.max(np.abs(ax[6].get_ylim()))])
    # Choose higher one
    lim = max(xlim, ylim)

    ax[6].set_xlim(-lim, lim)
    ax[6].set_ylim(-lim, lim)
    return f
