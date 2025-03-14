"""
Figure 1
"""

from .common import getSetup, subplotLabel
from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
    anndata_lrp_overlap,
)


def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)

    X = import_balf_covid()
    df_lrp = import_ligand_receptor_pairs()
    X, df_lrp = anndata_lrp_overlap(X, df_lrp)

    return f
