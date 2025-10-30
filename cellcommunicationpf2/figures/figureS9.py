"""
Figure S6: CCC-RISE on BALF alad data. Showing ligand-receptor factors.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import anndata
from .commonFuncs.plotFactors import plot_lr_factors_partial


def makeFigure():
    ax, f = getSetup((12, 12), (6, 6))
    subplotLabel(ax)

    X = anndata.read_h5ad("cellcommunicationpf2/alad_stable.h5ad")

    for i in range(X.uns["A"].shape[1]):
        plot_lr_factors_partial(X, i + 1, ax[2 * i], geneAmount=10, top=True)
        plot_lr_factors_partial(X, i + 1, ax[2 * i + 1], geneAmount=10, top=False)

    return f
