"""
Figure A3e: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import pandas as pd
import seaborn as sns

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 6
    
    X_mdc_sender = X[X.obs["celltype"] == "mDC"]
    X_mdc_sender = add_obs_cmp_label(X_mdc_sender, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="sender")
    X_mdc_sender = add_obs_cmp_unique_one(X_mdc_sender, cmp=ccc_rise_cmp)
    X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp-1])]

    X_mdc_receiver = X[(X.obs["celltype"] == "mDC")]
    X_mdc_receiver = add_obs_cmp_label(X_mdc_receiver, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="receiver")
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, "CCL19", "CCR7")
    sns.heatmap(df, ax=ax[0], cmap="viridis")

    X_b_receiver = X[(X.obs["celltype"] == "B")]
    X_b_receiver = add_obs_cmp_label(X_b_receiver, cmp=ccc_rise_cmp, pos=True, top_perc=25, type="receiver")
    X_b_receiver = add_obs_cmp_unique_one(X_b_receiver, cmp=ccc_rise_cmp)
    X_b_receiver = X_b_receiver[X_b_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_b_receiver = X_b_receiver[np.argsort(X_b_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]

    df = expression_product_matrix(X_mdc_sender, X_b_receiver, "CCL19", "CCR7")
    sns.heatmap(df, ax=ax[1], cmap="viridis")
    
    ax[0].set_xlabel("Receiver Cells (mDCs)")
    ax[0].set_ylabel("Sender Cells (mDCs)")
    ax[1].set_xlabel("Receiver Cells (B cells)")
    ax[1].set_ylabel("Sender Cells (mDCs)")
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