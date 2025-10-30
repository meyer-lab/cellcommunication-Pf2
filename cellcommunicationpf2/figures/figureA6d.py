"""
Figure A4d: Logistic regression weights for ALAD vs Control classification based on CCC-RISE components.
"""

import numpy as np
from ..import_data import add_cond_idxs
import anndata
from .common import getSetup, subplotLabel
from ..import_data import add_cond_idxs
from ..logreg import ccc_rise_logreg_weights
from ..utils import correct_conditions

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    X = anndata.read_h5ad("cellcommunicationpf2/alad_stable.h5ad")

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    sample_to_group = sample_to_group.loc[
        np.unique(X.obs[condition_column], return_index=True)[0]
    ]
    sample_to_group = sample_to_group.apply(
        lambda x: "declined" if x == "declined" else "stable"
    )
    sample_to_group = sample_to_group.astype("category").cat.codes
    
    X.uns["A"] = correct_conditions(X)

    auc_roc_weights, _ = ccc_rise_logreg_weights(X, sample_to_group)

    ax[0].bar(np.arange(len(auc_roc_weights))+1, auc_roc_weights)
    
    print(auc_roc_weights)

    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("Logistic Regression Weight")

    ym = np.max(np.abs(auc_roc_weights)) * 1.1
    ax[0].set_ylim(-ym, ym)
    ax[0].set_xticks(np.arange(len(auc_roc_weights))+1)

    return f
