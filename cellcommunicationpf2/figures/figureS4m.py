"""
Figure S4m: Violin plot of component 6 values for receiver cells in CCC-RISE on BALF COVID-19 data.
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
)
from .commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 6

    X_mdc_receiver = add_obs_cmp_label(
        X, cmp=ccc_rise_cmp, pos=True, top_perc=1, type="receiver"
    )
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]
    print("Receiver cells:", X_mdc_receiver.shape)
    
    # Datafame of the component values for the receiver cells and their cell types
    df = X_mdc_receiver.obs[["celltype"]]
    print(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])

    df["receiver"] = np.array(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    print(df)
    sns.violinplot(data=df, x="celltype", y="receiver", ax=ax[1])
    rotate_xaxis(ax[1], 90)
    

    return f
