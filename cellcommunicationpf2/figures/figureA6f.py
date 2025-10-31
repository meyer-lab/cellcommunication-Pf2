"""
Figure A5k_l: Heatmaps of ligand-receptor expression products for NK cell to Dendritic Cell interactions.
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

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad_stable.h5ad")
    ccc_rise_cmp = 5

    X_mdc_sender = X[X.obs["broad_cell_type"] == "CD8 T cells"]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_sender) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_sender), size=300, replace=False)
        X_mdc_sender = X_mdc_sender[sample_indices]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "CD8 T cells")]
    
    # Sample from whole cell population instead of filtering by top percentile
    if len(X_mdc_receiver) > 500:  # Limit sample size for computational efficiency
        sample_indices = np.random.choice(len(X_mdc_receiver), size=300, replace=False)
        X_mdc_receiver = X_mdc_receiver[sample_indices]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["GZMA", "F2R"]]
    for i, (lig, rec) in enumerate(pairs):
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        sns.heatmap(df, ax=ax[i], cmap="viridis")

        ax[i].set_xlabel("NK cells")
        ax[i].set_ylabel("NK Cells")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    return f