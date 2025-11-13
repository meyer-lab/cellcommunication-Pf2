"""
Figure A3e: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 6
    
    X_mdc_sender = X[X.obs["celltype"] == "mDC"]
    X_mdc_sender = X_mdc_sender[np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp-1])]

    X_mdc_receiver = X[(X.obs["celltype"] == "mDC")]


    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, "CCL19", "CCR7")    
    df = group_matrix(df)
    
    sns.heatmap(df, ax=ax[0], cmap="viridis", vmax=0.12)

    X_b_receiver = X[(X.obs["celltype"] == "B")]
    # Alter order based on factor value low to high
    X_b_receiver = X_b_receiver[np.argsort(X_b_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc_sender, X_b_receiver, "CCL19", "CCR7")
    df = group_matrix(df)
    
    sns.heatmap(df, ax=ax[1], cmap="viridis", vmax=0.12)

    ax[0].set_xlabel("Receiver Cells (mDCs)")
    ax[0].set_ylabel("Sender Cells (mDCs)")
    ax[1].set_xlabel("Receiver Cells (B cells)")
    ax[1].set_ylabel("Sender Cells (mDCs)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_title("CCL19-CCR7 Interaction")
    ax[1].set_title("CCL19-CCR7 Interaction")
    
    
    return f

