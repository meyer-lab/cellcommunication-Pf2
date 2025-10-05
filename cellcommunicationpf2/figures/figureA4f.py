"""
Figure A4f: Reading Seurat RDS files and RISE decomposition visualization.
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

    # Read the celltype CSV file
    celltypes = pd.read_csv("celltype.csv", index_col=0)  # Use index_col=0 since cell names are in the index
    X.obs["cell_type"] = celltypes.loc[X.obs.index, "celltype"]
    
    print("Original cell types:")
    print(X.obs["cell_type"].unique())
    
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
    X.obs["broad_cell_type"] = X.obs["cell_type"].apply(categorize_cell_type)


    # # Plot the different categorizations
    # plot_labels_pacmap(X, labelType="cell_type", ax=ax[0, 0])
    # ax[0, 0].set_title("Original Cell Types")
    
    plot_labels_pacmap(X, labelType="broad_cell_type", ax=ax[0])
    ax[0].set_title("Broad Cell Types")
    
    # plot_labels_pacmap(X, labelType="combined_category", ax=ax[1, 0])
    # ax[1, 0].set_title("Combined Category")

    plot_labels_pacmap(X, labelType="ALADstatus", ax=ax[1])
    ax[1].set_title("ALAD Status")

    return f
