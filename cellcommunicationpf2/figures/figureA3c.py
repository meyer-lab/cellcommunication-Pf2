"""
Figure A3c: CCC-RISE on BALF COVID-19 data. Showing weighted sender and receiver cell factors.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import anndata
from .commonFuncs.plotPaCMAP import (
    plot_wc_per_celltype,
    plot_wc_pacmap
)   

def makeFigure():
    ax, f = getSetup((18, 18), (5, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    cp_rank = X.uns["A"].shape[1]
            
    for i in range(cp_rank):
        plot_wc_per_celltype(X, i + 1, ax[i], cellType="celltype", factor_matrix="B")
        
    for i in range(cp_rank):
        plot_wc_per_celltype(X, i + 1, ax[i+cp_rank], cellType="celltype", factor_matrix="C")
        
    # for i in range(cp_rank):
    #     plot_wc_pacmap(X, i + 1, ax[i], factor_matrix="B", cbarMax=0.3)
        
    # for i in range(cp_rank):
    #     plot_wc_pacmap(X, i + 1, ax[i+cp_rank], factor_matrix="C", cbarMax=0.3)

    return f