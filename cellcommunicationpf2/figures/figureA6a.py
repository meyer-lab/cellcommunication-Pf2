"""
Figure A6a: RISE Rank Analysis on ALAD data distinguishing 'declined' vs 'stable' conditions.
"""

import numpy as np
from ..import_data import add_cond_idxs, import_alad
from .common import getSetup, subplotLabel
import numpy as np
from ..import_data import add_cond_idxs
from ..logreg import rise_ranks_logreg
import pandas as pd


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    print(X)

    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    sample_to_group = sample_to_group.loc[
        np.unique(X.obs[condition_column], return_index=True)[0]
    ]
     # Combine 'recovered' and 'control' into 'control', 'declined' into its own category
    sample_to_group = sample_to_group.apply(
        lambda x: "declined" if x == "declined" else "stable"
    )
    print(sample_to_group)
    sample_to_group = sample_to_group.astype("category").cat.codes
    print(sample_to_group)
    scoring = ["roc_auc", "accuracy"]

    # rank_list = list(np.append([1], range(5, 66, 5)))
    # # rank_list = list(range(1, 4, 2))

    # scores_aucroc, scores_accuracy = rise_ranks_logreg(
    #     X, rank_list, sample_to_group, scoring, n_iter_max=10000, tolerance=1e-9
    # )

    # ax[0].plot(rank_list, scores_aucroc)
    # ax[0].set_xlabel("RISE Rank")
    # ax[0].set_ylabel("10-Fold CV: roc_auc")
    # ax[0].set_ylim(0, np.max(scores_aucroc) + 0.05)

    # ax[1].plot(rank_list, scores_accuracy)
    # ax[1].set_xlabel("RISE Rank")
    # ax[1].set_ylabel("10-Fold CV: accuracy")
    # ax[1].set_ylim(0, np.max(scores_accuracy) + 0.05)

    return f
