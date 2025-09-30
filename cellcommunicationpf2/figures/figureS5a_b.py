"""
Figure S5a_b: Decomposition of pseudobulk communication data and tensor for BAL data
"""

from ..import_data import (
    add_cond_idxs,
    import_alad,
)
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_diff_ranks
import numpy as np

def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    print(X)

    # Add condition indices using dsco_id as the condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    # Parameters for stability plots
    rank_list = list(np.append([1], range(5, 66, 5)))
    rank_list = list(range(1, 11, 5))
    runs = 3
    runs = 1

    print("Plotting FMS vs. rank...")
    plot_fms_r2x_diff_ranks(X, condition_column, ax[0], ax[1], ranksList=rank_list, runs=runs)
    ax[0].set_title(f"RISE on COVID-19 scRNA-seq: {X.shape[1]} genes")
    ax[1].set_title(f"RISE on COVID-19 scRNA-seq: {X.shape[1]} genes")


    return f

