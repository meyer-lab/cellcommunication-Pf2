"""
Figure A4c: Violin plots of cell weight distributions for specific cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import anndata
import seaborn as sns

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    ccc_rise_cmp = 6
    
    # Violin plot of cell weighting distribution for Mast cells for a component
    # Keep only cells with mast cells for the violin plot
    X_mdc = X[X.obs["celltype"] == "mDC"]
    X_mdc = X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1]

    sns.violinplot(data=X_mdc, ax=ax[0])
    ax[0].set_ylim(-0.1, 0.85)
    ax[0].set_xlabel("mDC Weight Distribution")
    ax[0].set_ylabel("Sender Cell Component Association")
    
    ccc_rise_cmp = 5
    
    # Violin plot of cell weighting distribution for Mast cells for a component
    # Keep only cells with mast cells for the violin plot
    X_epithelial = X[X.obs["celltype"] == "Epithelial"]
    print(X_epithelial)
    X_epithelial_send = X_epithelial.obsm["sc_B"][:, ccc_rise_cmp-1]

    sns.violinplot(data=X_epithelial_send, ax=ax[1])
    ax[1].set_ylim(-0.15, 0.5)
    ax[1].set_xlabel("Epithelial Cell Weight Distribution")
    ax[1].set_ylabel("Sender Cell Component Association")
    
    
    X_epithelial_rec = X_epithelial.obsm["rc_C"][:, ccc_rise_cmp-1]

    sns.violinplot(data=X_epithelial_rec, ax=ax[2])
    ax[2].set_ylim(-0.15, 0.5)
    ax[2].set_xlabel("Epithelial Cell Weight Distribution")
    ax[2].set_ylabel("Receiver Cell Component Association")
    
    
    return f


