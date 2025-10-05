"""
Figure A4de: RISE decomposition with different ranks and PaCMAP projections visualization.
"""

import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
)
from ..tensor import rise_store_r2x
from ..import_data import import_alad
from parafac2.parafac2 import parafac2_nd, anndata_to_list
from pacmap import PaCMAP


def makeFigure():
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")

    X = import_alad(gene_threshold=0.001, normalize=True)
    print(X)
    

    rank = 15
    print(f"Running RISE with rank {rank}")
    gene_names = list(X.var_names)
    X_list = anndata_to_list(X)

    pf2_output, _ = parafac2_nd(
        X, rank=rank, n_iter_max=10000,random_state=0
    )
    _, _, projections = pf2_output

    projected_matrices = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        # Convert tensor to NumPy
        tensor_np = tensor.get()
        projected_matrices.append(proj.T @ tensor_np)
        
    sg_index = X.obs["condition_unique_idxs"]
    X.obsm["projections"] = np.zeros((X.shape[0], rank))
    for i, p in enumerate(projections):
            X.obsm["projections"][sg_index == i, :] = p

    # for i, resolution in enumerate([2, 3, 5, 6, 8, 10, 15, 25, 50, 100, 200]):
    #     pcm = PaCMAP(random_state=0, n_neighbors=resolution)
    #     X.obsm["PaCMAP"] = pcm.fit_transform(X.obsm["projections"])
    
    
    pcm = PaCMAP(random_state=0, n_neighbors=10)
    X.obsm["PaCMAP"] = pcm.fit_transform(X.obsm["projections"])
    

    plot_labels_pacmap(X, labelType="ALADstatus", ax=ax[0])

    XX = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    
    XX.obsm["PaCMAP"] = pcm.fit_transform(X.obsm["projections"])
    
    
    celltypes = pd.read_csv("celltype.csv", index_col=0)  # Use index_col=0 since cell names are in the index
    XX.obs["cell_type"] = celltypes.loc[XX.obs.index, "celltype"]

    # Create broader cell type categories
    def categorize_cell_type(cell_type):
        if pd.isna(cell_type):
            return "Other"
        
        cell_type = str(cell_type)
        
        # Macrophages (all types)
        if "Macrophage" in cell_type or "macrophage" in cell_type or "Mono/Mac" in cell_type:
            return "Macrophages"
        
        # Dendritic Cells
        elif any(dc_type in cell_type for dc_type in ["cDC1", "cDC2", "Activated DCs"]):
            return "Dendritic Cells"
        
        # Monocytes
        elif "Monocyte" in cell_type:
            return "Monocytes"
        
        # CD8 T cells
        elif "CD8" in cell_type:
            return "CD8 T cells"
        
        # CD4 T cells and Tregs
        elif "CD4" in cell_type or "Treg" in cell_type:
            return "CD4 T cells"
        
        # NK cells
        elif "NK" in cell_type:
            return "NK cells"
        
        # Proliferating cells
        elif "Proliferating" in cell_type:
            return "Proliferating cells"
        
        # Epithelial cells
        elif "epithelium" in cell_type:
            return "Epithelial cells"
        
        # Everything else
        else:
            return "Other"
    
    # Apply the categorization
    XX.obs["broad_cell_type"] = XX.obs["cell_type"].apply(categorize_cell_type)


    
    plot_labels_pacmap(XX, labelType="broad_cell_type", ax=ax[1])
    
    X.obs["broad_cell_type"] = XX.obs["broad_cell_type"]

    plot_labels_pacmap(X, labelType="broad_cell_type", ax=ax[2])
    
    
    
    # XX.write_h5ad("bal_alad.h5ad")
    
    return f