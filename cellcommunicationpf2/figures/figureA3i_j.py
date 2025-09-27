"""
Figure A3i-j: Violin plots of cell weight distributions for specific cell types and components in CCC-RISE on BALF COVID-19 data.
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
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    
    ccc_rise_cmp = 5
    
    
    # Violin plot of cell weighting distribution for Mast cells for a component
    # Keep only cells with mast cells for the violin plot
    X_epithelial = X[X.obs["celltype"] == "Epithelial"]
    X_epithelial_send = X_epithelial.obsm["sc_B"][:, ccc_rise_cmp-1]

    sns.violinplot(data=X_epithelial_send, ax=ax[1])
    ax[1].set_ylim(-0.15, 0.5)
    ax[1].set_xlabel("Epithelial Cell Weight Distribution")
    ax[1].set_ylabel(f"Sender Cell Component {ccc_rise_cmp} Association")
    
    
    X_epithelial_rec = X_epithelial.obsm["rc_C"][:, ccc_rise_cmp-1]

    sns.violinplot(data=X_epithelial_rec, ax=ax[2])
    ax[2].set_ylim(-0.15, 0.5)
    ax[2].set_xlabel("Epithelial Cell Weight Distribution")
    ax[2].set_ylabel(f"Receiver Cell Component {ccc_rise_cmp} Association")
    
    
    return f


