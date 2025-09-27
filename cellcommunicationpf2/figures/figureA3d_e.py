"""
Figure A3d_e: Violin plots of cell weight distributions for specific cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
import pandas as pd
import numpy as np


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 3
    
    # Violin plot of cell weighting distribution for Mast cells for a component
    # Keep only cells with mast cells for the violin plot
    both = ["sc_B", "rc_C"]
    for i, b in enumerate(both):
        X_mdc = X[X.obs["celltype"] == "Epithelial"]
        X_mdc = X_mdc.obsm[b][:, ccc_rise_cmp-1]
        if b == "sc_B":
            title = "Sender"
        else:
            title = "Receiver"
        # Order from low to high weights
        X_mdc = X_mdc[np.argsort(X_mdc)]

        # Split into groups top 1, top 1-5, top 5-10, top 10-30, 30-50, 50-70, 70-90, bottom 10
        thresholds = np.percentile(X_mdc, [10, 30, 50, 70, 90, 95, 99])
        X_mdc_split = np.empty(X_mdc.shape, dtype=object)
        X_mdc_split[X_mdc <= thresholds[0]] = '0-10%'
        X_mdc_split[(X_mdc > thresholds[0]) & (X_mdc <= thresholds[1])] = '10-30%'
        X_mdc_split[(X_mdc > thresholds[1]) & (X_mdc <= thresholds[2])] = '30-50%'
        X_mdc_split[(X_mdc > thresholds[2]) & (X_mdc <= thresholds[3])] = '50-70%'
        X_mdc_split[(X_mdc > thresholds[3]) & (X_mdc <= thresholds[4])] = '70-90%'
        X_mdc_split[(X_mdc > thresholds[4]) & (X_mdc <= thresholds[5])] = '90-95%'
        X_mdc_split[(X_mdc > thresholds[5]) & (X_mdc <= thresholds[6])] = '95-99%'
        X_mdc_split[X_mdc > thresholds[6]] = '99-100%'
        
        X_mdc_split = pd.DataFrame({'Epithelial Weight': X_mdc, 'Group': X_mdc_split})

        sns.violinplot(data=X_mdc_split, ax=ax[i],  x='Group', y="Epithelial Weight", order=['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-95%', '95-99%', '99-100%'])
        ax[i].set_xlabel("Epithelial Weight Distribution")
        ax[i].set_ylabel(f"{title} Cell Component Association")
        ax[i].tick_params(axis='x', rotation=45)

    return f

