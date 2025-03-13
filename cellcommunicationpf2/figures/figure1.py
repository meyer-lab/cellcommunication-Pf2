"""
Figure 1
"""

from .common import getSetup, subplotLabel
from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
    anndata_lrp_overlap,
)
from ..ccc import calc_communication_score


def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)

    X = import_balf_covid()
    df_lrp = import_ligand_receptor_pairs()
    X, df_lrp = anndata_lrp_overlap(X, df_lrp)

    # Make smaller dataset for now
    X = X[::200]
    df_lrp = df_lrp.iloc[:20, :]

    Xccc = calc_communication_score(X, df_lrp, communication_score="expression_product")

    return f
