"""
Figure A5d: CCC-RISE on BALF COVID-19 data. Showing PaCMAP of cells colored by cell type.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
)   
from pacmap import PaCMAP
import pandas as pd

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
import seaborn as sns

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    ccc_rise_cmp = 6
    
    # # Violin plot of cell weighting distribution for Mast cells for a component
    # # Keep only cells with mast cells for the violin plot
    # X_mdc = X[X.obs["celltype"] == "mDC"]
    # print(X_mdc)
    # X_mdc = X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1]

    # sns.violinplot(data=X_mdc, ax=ax[0])
    # ax[0].set_ylim(-0.1, 0.85)
    # ax[0].set_xlabel("mDC Weight Distribution")
    # ax[0].set_ylabel("Sender Cell Component Association")
    
    
    # ccc_rise_cmp = 5
    
    # # Violin plot of cell weighting distribution for Mast cells for a component
    # # Keep only cells with mast cells for the violin plot
    # X_epithelial = X[X.obs["celltype"] == "Epithelial"]
    # print(X_epithelial)
    # X_epithelial_send = X_epithelial.obsm["sc_B"][:, ccc_rise_cmp-1]

    # sns.violinplot(data=X_epithelial_send, ax=ax[1])
    # ax[1].set_ylim(-0.15, 0.5)
    # ax[1].set_xlabel("Epithelial Cell Weight Distribution")
    # ax[1].set_ylabel("Sender Cell Component Association")
    
    
    # X_epithelial_rec = X_epithelial.obsm["rc_C"][:, ccc_rise_cmp-1]

    # sns.violinplot(data=X_epithelial_rec, ax=ax[2])
    # ax[2].set_ylim(-0.15, 0.5)
    # ax[2].set_xlabel("Epithelial Cell Weight Distribution")
    # ax[2].set_ylabel("Receiver Cell Component Association")
    
    
    
    X = anndata.read_h5ad("cellcommunicationpf2/data/bal/bal_updated.h5ad")
    X_mdc = X[X.obs["celltype"] == "mDC"]
    print(X_mdc)
    
    
    X_mdc = add_obs_cmp_label(X_mdc, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="sender")
    X_mdc = add_obs_cmp_unique_one(X_mdc, cmp=ccc_rise_cmp)

    X_mdc = X_mdc[X_mdc.obs["Label"] != "NoLabel"]
    print(X_mdc)
    
    # Alter order based on factor value high to low
    print(np.argsort(-X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1]))

    X_mdc = X_mdc[np.argsort(-X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1])]
    
    print(np.argsort(-X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1]))
    print(X_mdc.obsm["sc_B"][:, ccc_rise_cmp-1])




    X_both = X[(X.obs["celltype"] == "mDC")]
    print(X_both)
    
    X_both = add_obs_cmp_label(X_both, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="receiver")
    X_both = add_obs_cmp_unique_one(X_both, cmp=ccc_rise_cmp)
    X_both = X_both[X_both.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_both = X_both[np.argsort(X_both.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc, X_both, "CCL19", "CCR7")


    
    # Plot heatmap of expression product matrix
    sns.heatmap(df, ax=ax[0], cmap="viridis")
    ax[0].set_xlabel("Receiver Cells (mDCs)")
    ax[0].set_ylabel("Sender Cells (mDCs)")
    
    
    
    X_b = X[(X.obs["celltype"] == "B")]
    print(X_b)

    X_b = add_obs_cmp_label(X_b, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="receiver")
    X_b = add_obs_cmp_unique_one(X_b, cmp=ccc_rise_cmp)
    X_b = X_b[X_b.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_b = X_b[np.argsort(X_b.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc, X_b, "CCL19", "CCR7")


    
    # Plot heatmap of expression product matrix
    sns.heatmap(df, ax=ax[1], cmap="viridis")
    ax[1].set_xlabel("Receiver Cells (B cells)")
    ax[1].set_ylabel("Sender Cells (mDCs)")
    
    # no row/column labels
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_title("CCL19-CCR7 Interaction")
    ax[1].set_title("CCL19-CCR7 Interaction")
    return f


def add_obs_cmp_label(
    X: anndata.AnnData, cmp: int, pos: bool = True, top_perc: float = 1, type: str = "receiver"
):
    """Adds a boolean label to X.obs for cells in the top or bottom percentage of a single component."""
    if type == "sender":
        factor_type = X.obsm["sc_B"]
    elif type == "receiver":
        factor_type = X.obsm["rc_C"]
  
    if pos:
        thres_value = 100 - top_perc
        threshold = np.percentile(factor_type, thres_value, axis=0)
        idx = factor_type[:, cmp - 1] > threshold[cmp - 1]
    else:
        thres_value = top_perc
        threshold = np.percentile(factor_type, thres_value, axis=0)
        idx = factor_type[:, cmp - 1] < threshold[cmp - 1]

    X.obs[f"Cmp{cmp}"] = idx
    
    return X


def add_obs_cmp_unique_one(X: anndata.AnnData, cmp: str):
    """Creates AnnData observation column for a single component."""
    label_col = f"Cmp{cmp}"
    X.obs["Label"] = np.where(X.obs[label_col], label_col, "NoLabel")
    
    return X


def expression_product_matrix(X1, X2, ligand, receptor):
    """
    For each cell in X1 and each cell in X2, compute the product:
    X1[cell_i, ligand] * X2[cell_j, receptor]
    Returns a DataFrame with X1 cells as rows and X2 cells as columns.
    """
    # Ensure gene names are present
    assert ligand in X1.var_names, f"{ligand} not in X1"
    assert receptor in X2.var_names, f"{receptor} not in X2"
        
    # Get expression vectors

    # Ensure 1D dense arrays
    # Convert to dense 1D arrays, even if sparse
    expr1 = X1[:, ligand].X
    if hasattr(expr1, 'toarray'):
        expr1 = expr1.toarray().flatten()
    else:
        expr1 = np.ravel(np.array(expr1))

    expr2 = X2[:, receptor].X
    if hasattr(expr2, 'toarray'):
        expr2 = expr2.toarray().flatten()
    else:
        expr2 = np.ravel(np.array(expr2))
        
    # Compute outer product
    product_matrix = np.outer(expr1, expr2)

    # Build DataFrame
    df = pd.DataFrame(
        product_matrix,
        index=X1.obs_names,
        columns=X2.obs_names
    )
    return df