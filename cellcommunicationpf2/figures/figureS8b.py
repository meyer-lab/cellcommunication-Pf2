"""
Figure S1e_g: Cell type proportions across conditions in BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr


def makeFigure():
    ax, f = getSetup((12, 6), (1, 2))  # Create 1x2 subplot layout
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    sample_id = "dsco_id"
    condition_id = "ALADstatus"
    celltype_id = "broad_cell_type"

    df = X.obs.groupby([celltype_id, sample_id], observed=False).size().reset_index(name="count")

    # Add condition information back by merging with sample-condition mapping
    sample_condition_map = X.obs[[sample_id, condition_id]].drop_duplicates()
    df = df.merge(sample_condition_map, on=sample_id, how="left")

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
    df[condition_id] = df[condition_id].astype(str).replace({"recovered": "ALAD", "declined": "ALAD"})

    # Get unique ALAD statuses
    alad_statuses = df[condition_id].unique()
    print(f"ALAD statuses: {alad_statuses}")
    
    # Create separate heatmaps for each ALAD status
    for i, status in enumerate(alad_statuses):
        # Filter data for this ALAD status
        df_status = df[df[condition_id] == status]
        
        # Create pivot table for this status
        proportion_pivot = df_status.pivot_table(
            index=sample_id, columns=celltype_id, values="proportion", fill_value=0, observed=False
        )
        
        print(f"\n{status} - Proportion pivot shape: {proportion_pivot.shape}")
        
        # Calculate correlation p-values
        p_values = proportion_pivot.corr(method=lambda x, y: pearsonr(x, y).pvalue)
            
        # Only plot the lower triangle of the heatmap, including the diagonal
        mask = np.triu(np.ones_like(p_values, dtype=bool), k=1)

        sns.heatmap(p_values, mask=mask,
                    cmap='coolwarm', 
                    ax=ax[i], 
                    cbar_kws={'label': 'Pearson correlation p-value'},
                    vmin=0, vmax=.05)
        
        ax[i].set_title(f'Cell Type Correlation p-values - {status}')
        ax[i].set_xlabel('Cell Type')
        ax[i].set_ylabel('Cell Type')
    
    return f
