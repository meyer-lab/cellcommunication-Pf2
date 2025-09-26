"""
Figure A5b: Correlation of condition factors with disease severity in CCC-RISE on BAL ALAD data
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
import pandas as pd
import numpy as np
from ..utils import correct_conditions
from scipy.stats import pearsonr


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    print(X.X.dtype)
    XX = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    print(XX.X.dtype)
    
    # Check if the data is sparse
    print(f"Is X sparse? {X.X.format if hasattr(X.X, 'format') else 'No'}")
    print(f"Is XX sparse? {XX.X.format if hasattr(XX.X, 'format') else 'No'}")
    
    
    # condition_column = "dsco_id"
    # X.uns["A"] = correct_conditions(X)

    # group_col = "ALADstatus"
    # sample_to_group = X.obs.drop_duplicates(
    #     subset=[condition_column, group_col]
    # ).set_index(condition_column)[group_col]
    
    # severity_map = {k: i+1 for i, k in enumerate(np.unique(sample_to_group))}
    # severity_numeric = sample_to_group.map(severity_map).values

    # cmp = X.uns["A"].shape[1]
    # pearson_corr = np.zeros((cmp))
    # for i in range(cmp):
    #     pearson_corr[i] = pearsonr(X.uns["A"][:, i-1], severity_numeric)[0]
    # pearson_df = pd.DataFrame(pearson_corr, index=[f"{i+1}" for i in range(cmp)], columns=["Pearson Correlation"])
    # pearson_df["Component"] = pearson_df.index
    # print(pearson_df)
    # sns.barplot(x="Component", y="Pearson Correlation", data=pearson_df, ax=ax[0], color="black")
    # # ax[0].set_ylim(-0.1, 1)
   
    
   
    return f


