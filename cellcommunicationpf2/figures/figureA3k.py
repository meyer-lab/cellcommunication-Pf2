"""
Figure A3k: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
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
    average_product_matrix_ccc
    
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 5
    
    X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender = X_epi_sender[
        np.argsort(-X_epi_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_epi_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_epi_receiver = X_epi_receiver[
        np.argsort(X_epi_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]
    
    pairs = [["CDH1", "CDH1"], ["OCLN", "OCLN"], ["PRSS3", "F2RL1"]]
    for i, (lig, rec) in enumerate(pairs):
        df = expression_product_matrix(X_epi_sender, X_epi_receiver, lig, rec)
        df = average_product_matrix_ccc(df)
        sns.heatmap(df, ax=ax[i], cmap="viridis")
        ax[i].set_xlabel("Receiver Epithelial Cells")
        ax[i].set_ylabel("Sender Epithelial Cells")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    return f

