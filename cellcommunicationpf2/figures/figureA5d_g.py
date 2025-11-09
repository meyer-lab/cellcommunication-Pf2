"""
Figure A5d_g: Violin plots of cell weight distributions for specific cell types and components in CCC-RISE on BALF ALAD data.
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
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")

    ccc_rise_cmp = 13

    # Violin plot of cell weighting distribution for Dendritic Cells for a component
    # Keep only cells with Dendritic Cells for the violin plot
    X_dc = X[X.obs["broad_cell_type"] == "Dendritic Cells"]
    X_dc_send = X_dc.obsm["sc_B"][:, ccc_rise_cmp - 1]

    sns.violinplot(data=X_dc_send, ax=ax[0])
    ax[0].set_ylim(-1.0, 0.4)
    ax[0].set_xlabel("Dendritic Cells Weight Distribution")
    ax[0].set_ylabel(f"Sender Cell Component {ccc_rise_cmp} Association")

    X_dc_rec = X_dc.obsm["rc_C"][:, ccc_rise_cmp - 1]

    sns.violinplot(data=X_dc_rec, ax=ax[1])
    ax[1].set_ylim(-1.0, 0.4)
    ax[1].set_xlabel("Dendritic Cells Weight Distribution")
    ax[1].set_ylabel(f"Receiver Cell Component {ccc_rise_cmp} Association")

    X_epithelial = X[X.obs["broad_cell_type"] == "Epithelial cells"]
    X_epithelial_rec = X_epithelial.obsm["rc_C"][:, ccc_rise_cmp - 1]

    sns.violinplot(data=X_epithelial_rec, ax=ax[3])
    ax[3].set_ylim(-1.0, 0.4)
    ax[3].set_xlabel("Epithelial cells Weight Distribution")
    ax[3].set_ylabel(f"Receiver Cell Component {ccc_rise_cmp} Association")

    return f