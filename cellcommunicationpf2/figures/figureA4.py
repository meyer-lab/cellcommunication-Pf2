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
import pandas as pd
import seaborn as sns

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
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
        lambda x: "alad" if x != "control" else "control"
    )
    sample_to_group = sample_to_group.astype("category").cat.codes
    
    X.uns["A"] = correct_conditions(X)

    
    
    sample_id = "dsco_id"
    condition_id = "ALADstatus"
    celltype_id = "broad_cell_type"

    df = X.obs.groupby([celltype_id, sample_id], observed=False).size().reset_index(name="count")

    # Add ALADstatus (condition) directly to df using a map from sample_id to condition
    sample_to_condition = X.obs.drop_duplicates(subset=[sample_id]).set_index(sample_id)[condition_id]
    df[condition_id] = df[sample_id].map(sample_to_condition)

    df[celltype_id] = pd.Categorical(
        df[celltype_id], categories=np.unique(df[celltype_id]), ordered=True
    )
    df[sample_id] = pd.Categorical(
        df[sample_id], categories=np.unique(df[sample_id]), ordered=True
    )
    df[condition_id] = pd.Categorical(
        df[condition_id], categories=np.unique(df[condition_id]), ordered=True
    )
    df = df.sort_values([condition_id, sample_id, celltype_id])
    df["proportion"] = df.groupby(sample_id, observed=False)["count"].transform(lambda x: x / x.sum())
    
    print(df)
    # Add X.uns information to df based on sample_id
    sample_factors = X.uns["A"]
    sample_categories = X.obs.drop_duplicates(subset=[sample_id]).set_index(sample_id)
    for i in range(sample_factors.shape[1]):
        df[f"Cmp. {i+1}"] = df[sample_id].map(
            lambda x: sample_factors[sample_categories.index.get_loc(x), i]
        )


    # Plot scatterplot  of CD4 T cell proportion with each CCC-RISE component 13
    target_components = 13  
    celltypes = ["CD8 T cells", "NK cells"]
    types = ["proportion", "count"]
    axs=0
    for i, celltype in enumerate(celltypes):
        for j, type_ in enumerate(types):
            comp = target_components
            
            sns.scatterplot(
                data=df[df[celltype_id] == celltype],
                x=f"Cmp. {comp}",
                y=type_,
                hue=condition_id,
                ax=ax[axs],
            )
            ax[axs].set_title(f"Component {comp} vs {celltype} {type_}")
            ax[axs].set_xlabel(f"Component {comp} Value")
            ax[axs].set_ylabel("CD4 T Cell Proportion")
            ax[axs].legend(title=condition_id)
            axs = axs + 1
            
   
    return f
