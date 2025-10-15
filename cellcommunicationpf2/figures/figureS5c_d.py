"""
Figure S5c_d: Decomposition of the communication tensor from Tensorcell2cell
"""

import seaborn as sns
import numpy as np
from ..import_data import add_cond_idxs, import_alad, import_ligand_receptor_pairs
from .common import getSetup, subplotLabel
from ..tensor import run_fms_r2x_analysis, calculate_interaction_tensor


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X_filtered = add_cond_idxs(X, condition_column)

    # Calculate interaction tensor
    interaction_tensor = calculate_interaction_tensor(
        X_filtered, lr_pairs, rise_rank=15
    )
    print("Interaction tensor shape:", interaction_tensor.shape)

    # Run FMS and R2X analysis
    # rank_list = list(range(1, 27, 1))
    # rank_list = list(range(1, 4, 2))
    # runs = 3
    # runs = 1
    # df = run_fms_r2x_analysis(
    #     interaction_tensor, rank_list=rank_list, runs=runs, svd_init="random"
    # )

    # sns.lineplot(data=df, x="Component", y="FMS", ax=ax[0], label="FMS")
    # ax[0].set_ylim(0, 1)

    # sns.lineplot(data=df, x="Component", y="R2X", ax=ax[1], color="orange", label="R2X")
    # ax[1].set_ylim(0, np.max(df["R2X"]) + 0.02)

    return f
