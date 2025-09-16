"""
Figure A5d: CCC-RISE on BALF COVID-19 data. Showing PaCMAP of cells colored by cell type.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
)   
from pacmap import PaCMAP

def makeFigure():
    ax, f = getSetup((12, 12), (4, 4))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal.h5ad")
    
    # Count how many cell in each cell type are present
    celltype_counts = X.obs['celltype'].value_counts()
    print("Cell type counts:\n", celltype_counts)
    
    n_neighbors = [1, 2, 3, 4, 5, 10, 15]
    for i, n in enumerate(n_neighbors):
        pcm = PaCMAP(n_neighbors=n)
        X.obsm["Pf2_PaCMAP"] = pcm.fit_transform(X.obsm["Pf2_projections"])

        plot_labels_pacmap(X, labelType="celltype", ax=ax[i])
        
        
        

    return f