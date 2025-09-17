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

from .common import (
    subplotLabel,
    getSetup,
)
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..utils import run_ccc_rise_workflow
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
import anndata
from .commonFuncs.plotGeneral import rotate_yaxis

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    
    # Count how many cell in each cell type are present
    celltype_counts = X.obs['celltype'].value_counts()
    print("Cell type counts:\n", celltype_counts)
    
    plot_labels_pacmap(X, labelType="celltype", ax=ax[0])
    plot_labels_pacmap(X, labelType="condition", ax=ax[1])
    plot_labels_pacmap(X, labelType="sample", ax=ax[2])

    return f