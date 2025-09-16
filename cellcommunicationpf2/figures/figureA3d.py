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

    # # Import Anndata file
    # X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    
    # # Count how many cell in each cell type are present
    # celltype_counts = X.obs['celltype'].value_counts()
    # print("Cell type counts:\n", celltype_counts)
    
    # plot_labels_pacmap(X, labelType="celltype", ax=ax[0])
    # plot_labels_pacmap(X, labelType="condition", ax=ax[1])
    # plot_labels_pacmap(X, labelType="sample", ax=ax[2])
        
    # Import and prepare data
    adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata, condition_column)

    # Create a mapping from each sample to its corresponding condition (e.g., 'severe')
    # This will be used for grouping and coloring the heatmap
    group_col = "condition"
    sample_to_group = adata_filtered.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

    # Parameters for CCC-RISE
    rise_rank = 70
    cp_rank = 8
    n_iter_max = 10000
    tol = 1e-9

    print(f"Running CCC-RISE with rank={rise_rank} and cp_rank={cp_rank}...")
    adata_filtered, _ = run_ccc_rise_workflow(
        adata_filtered,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        condition_column=condition_column,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        complex_sep="&",
    
    )
    
    plot_labels_pacmap(adata_filtered, labelType="celltype", ax=ax[0])
    plot_labels_pacmap(adata_filtered, labelType="condition", ax=ax[1])
    plot_labels_pacmap(adata_filtered, labelType="sample", ax=ax[2])

    return f