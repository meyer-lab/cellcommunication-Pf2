"""
Figure S2i_j: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 6
    print(X.obs["celltype"].unique())

    X_mdc_sender = X[X.obs["celltype"] == "Macrophages"]
    print("T cell sender cells:", X_mdc_sender.shape)
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_sender) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_sender), size=300, replace=False)
        X_mdc_sender = X_mdc_sender[sample_indices]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["celltype"] == "Macrophages")]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_receiver) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_receiver), size=300, replace=False)
        X_mdc_receiver = X_mdc_receiver[sample_indices]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, "CCL19", "CCR7")
    sns.heatmap(df, ax=ax[0], cmap="viridis", vmin=0, vmax=.001)

    X_b_receiver = X[(X.obs["celltype"] == "NK")]
    print("NK cell receiver shape:", X_b_receiver.shape)
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_b_receiver) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_b_receiver), size=300, replace=False)
        X_b_receiver = X_b_receiver[sample_indices]

    # Alter order based on factor value low to high
    X_b_receiver = X_b_receiver[
        np.argsort(X_b_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    df = expression_product_matrix(X_mdc_sender, X_b_receiver, "CCL19", "CCR7")
    sns.heatmap(df, ax=ax[1], cmap="viridis", vmin=0, vmax=.001)

    ax[0].set_xlabel("Receiver Cells (Macrophages)")
    ax[0].set_ylabel("Sender Cells (Macrophages)")
    ax[1].set_xlabel("Receiver Cells (NK cells)")
    ax[1].set_ylabel("Sender Cells (Macrophages)")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_title("CCL19-CCR7 Interaction")
    ax[1].set_title("CCL19-CCR7 Interaction")

    return f
