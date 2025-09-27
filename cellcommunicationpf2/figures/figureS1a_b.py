"""
Figure S1a_b: FMS across RISE ranks (only) for COVID-19
"""

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
)
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_diff_ranks


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_balf_covid(gene_threshold=0.001, normalize=True)

    # Add condition indices using sample as the condition
    condition_column = "sample"
    X = add_cond_idxs(X, condition_column)

    # Parameters for stability plots
    ranks = list(range(1, 61, 5))
    ranks = list(range(1, 11, 5))
    runs = 3
    runs = 1

    print("Plotting FMS vs. rank...")
    plot_fms_r2x_diff_ranks(X, condition_column, ax[0], ax[1], ranksList=ranks, runs=runs)
    ax[0].set_title(f"RISE on COVID-19 scRNA-seq: {X.shape[1]} genes")
    ax[1].set_title(f"RISE on COVID-19 scRNA-seq: {X.shape[1]} genes")


    return f

