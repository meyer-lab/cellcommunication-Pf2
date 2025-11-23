"""
Figure A4d: Logistic regression weights for ALAD vs Control classification based on CCC-RISE components.
"""

import numpy as np
from ..import_data import add_cond_idxs
import anndata
import pandas as pd
import itertools
import seaborn as sns
from matplotlib.axes import Axes
from .common import getSetup, subplotLabel
from ..import_data import add_cond_idxs
from ..utils import correct_conditions
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from .commonFuncs.plotGeneral import rotate_xaxis, rotate_yaxis

def makeFigure():
    ax, f = getSetup((6, 6), (1, 1))
    subplotLabel(ax)

    # Import and prepare data
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")

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
    print(np.unique(sample_to_group, return_counts=True))
    sample_to_group = sample_to_group.astype("category").cat.codes
    print(np.unique(sample_to_group, return_counts=True))
    
    
    print(sample_to_group.shape)
    print()

    pair_logistic_regression(correct_conditions(X), sample_to_group, ax[0])


    return f



def pair_logistic_regression(X: anndata.AnnData, status_df: pd.DataFrame, ax: Axes):
    """Plot factor weights for donor SLE prediction"""
    lrmodel = LogisticRegression(penalty=None, random_state=0)
    all_comps = np.arange(X.shape[1])
    acc = np.zeros((X.shape[1], X.shape[1]))

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] >= comps[1]:
            compFacs = X[:, [comps[0], comps[1]]]
            LR_CoH = lrmodel.fit(compFacs, status_df.values)
            acc[comps[0], comps[1]] = LR_CoH.score(compFacs, status_df.values)
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    mask = np.triu(np.ones_like(acc, dtype=bool))

    for i in range(len(mask)):
        mask[i, i] = False

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        data=acc,
        # vmin=0.5,
        # vmax=1,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        mask=mask,
        cbar_kws={"label": "Prediction Accuracy"},
        ax=ax,
        cmap=cmap,
    )

    ax.set(xlabel="Component", ylabel="Component")
    rotate_xaxis(ax, rotation=0)
    rotate_yaxis(ax, rotation=0)