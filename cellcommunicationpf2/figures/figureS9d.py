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
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    ccc_rise_cmp = 16

    X_mdc_sender = X[X.obs["broad_cell_type"] == "NK cells"]
 

    X_mdc_sender = add_obs_cmp_label(
        X_mdc_sender, cmp=ccc_rise_cmp, pos=True, top_perc=75, type="sender"
    )
    X_mdc_sender = add_obs_cmp_unique_one(X_mdc_sender, cmp=ccc_rise_cmp)
    X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]
    print("NK cells sender cells:", X_mdc_sender.shape)

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]
    print("Dendritic Cells receiver cells:", X_mdc_receiver.shape)
    X_mdc_receiver = add_obs_cmp_label(
        X_mdc_receiver, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="receiver"
    )
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]


    pairs = [["XCL2", "XCR1"], ["XCL1", "XCR1"]]
    for i, (lig, rec) in enumerate(pairs):
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        sns.heatmap(df, ax=ax[i], cmap="viridis")

        ax[i].set_xlabel("Receiver Dendritic Cells")
        ax[i].set_ylabel("Sender NK Cells")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    return f
