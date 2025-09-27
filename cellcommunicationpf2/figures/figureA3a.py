"""
Figure A3a: Comparison of CCC-RISE and Pseudobulk CPD on BALF COVID-19
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

def makeFigure():
    ax, f = getSetup((3, 3), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    # Reorder index based on np.unique
    sample_to_group = sample_to_group.loc[np.unique(X.obs[condition_column], return_index=True)[0]]
    A_factor = X.uns["A"]
    
    severity_map = {k: i+1 for i, k in enumerate(np.unique(sample_to_group))}
    severity_numeric = sample_to_group.map(severity_map).values

    cmp = A_factor.shape[1]
    pearson_corr = np.zeros((cmp))
    for i in range(cmp):
        pearson_corr[i] = pearsonr(A_factor[:, i], severity_numeric)[0]
    pearson_df = pd.DataFrame(pearson_corr, index=[f"{i+1}" for i in range(cmp)], columns=["Pearson Correlation"])
    pearson_df["Component"] = pearson_df.index
    sns.barplot(x="Component", y="Pearson Correlation", data=pearson_df, ax=ax[0], color="black")
    ax[0].set_ylim(-0.1, 1)
    

    return f