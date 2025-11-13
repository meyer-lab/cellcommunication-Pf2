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

    both = ["sc_B", "rc_C"]
    for i, b in enumerate(both):
        X_all_epithelial = X[X.obs["celltype"] == "Epithelial"]
        X_epi= X_all_epithelial.obsm[b][:, ccc_rise_cmp - 1]
        if b == "sc_B":
            title = "Sender"
        else:
            title = "Receiver"
        X_epi = X_epi[np.argsort(X_epi)]
        thresholds = np.percentile(X_epi, [10, 30, 50, 70, 90, 95, 99])
        X_epi_split = np.empty(X_epi.shape, dtype=object)
        X_epi_split[X_epi <= thresholds[0]] = "0-10%"
        X_epi_split[(X_epi > thresholds[0]) & (X_epi <= thresholds[1])] = "10-30%"
        X_epi_split[(X_epi > thresholds[1]) & (X_epi <= thresholds[2])] = "30-50%"
        X_epi_split[(X_epi > thresholds[2]) & (X_epi <= thresholds[3])] = "50-70%"
        X_epi_split[(X_epi > thresholds[3]) & (X_epi <= thresholds[4])] = "70-90%"
        X_epi_split[(X_epi > thresholds[4]) & (X_epi <= thresholds[5])] = "90-95%"
        X_epi_split[(X_epi > thresholds[5]) & (X_epi <= thresholds[6])] = "95-99%"
        X_epi_split[X_epi > thresholds[6]] = "99-100%"

        X_epi_split = pd.DataFrame({"Epithelial Weight": X_epi, "Group": X_epi_split})

        sns.violinplot(
            data=X_epi_split,
            ax=ax[i],
            x="Group",
            y="Epithelial Weight",
            order=[
                "0-10%",
                "10-30%",
                "30-50%",
                "50-70%",
                "70-90%",
                "90-95%",
                "95-99%",
                "99-100%",
            ],
        )
        ax[i].set_xlabel("Epithelial Weight Distribution")
        ax[i].set_ylabel(f"{title} Cell Component Association")
        ax[i].tick_params(axis="x", rotation=45)

    return f
