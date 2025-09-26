"""
Figure A3f: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)

def makeFigure():
    ax, f = getSetup((10, 10), (3, 3))  # 1 row, 3 columns for 3 L-R  pairs
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp1 = 3
    ccc_rise_cmp2 = 5
    
    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    X_mdc_sender = add_obs_cmp_both_label(X_mdc_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2, pos1=True, pos2=True, top_perc=10, type="sender")
    X_mdc_sender = add_obs_cmp_unique_two(X_mdc_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2)
    # X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_mdc_receiver = add_obs_cmp_both_label(X_mdc_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2, pos1=True, pos2=True, top_perc=10, type="receiver")
    X_mdc_receiver = add_obs_cmp_unique_two(X_mdc_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2)
    # X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    print("Epithelial sender cells:", X_mdc_sender.shape)
    print("Epithelial receiver cells:", X_mdc_receiver.shape)
    
   # Calculate amount of cells that fall into each label category and put in boxplot 
    label_counts = X_mdc_sender.obs['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    print("Sender cell label counts:\n", label_counts)
        
    sns.barplot(data=label_counts, x='Label', y='Count', ax=ax[0])
    ax[0].set_title("Sender Cell Label Counts")
    label_counts_receiver = X_mdc_receiver.obs['Label'].value_counts().reset_index()
    label_counts_receiver.columns = ['Label', 'Count']
    print("Receiver cell label counts:\n", label_counts_receiver)
    
    sns.barplot(data=label_counts_receiver, x='Label', y='Count', ax=ax[1])
    
    # Plot distribution of factor values for each label
    sns.boxplot(data=X_mdc_sender.obs, x='Label', y=X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp1-1], ax=ax[2])
    ax[2].set_title(f"Sender Factor {ccc_rise_cmp1} Distribution by Label")
    sns.boxplot(data=X_mdc_receiver.obs, x='Label', y=X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp1-1], ax=ax[3])
    ax[3].set_title(f"Receiver Factor {ccc_rise_cmp1} Distribution by Label")
    
    sns.boxplot(data=X_mdc_sender.obs, x='Label', y=X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp2-1], ax=ax[4])
    ax[4].set_title(f"Sender Factor {ccc_rise_cmp2} Distribution by Label")
    sns.boxplot(data=X_mdc_receiver.obs, x='Label', y=X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp2-1], ax=ax[5])
    ax[5].set_title(f"Receiver Factor {ccc_rise_cmp2} Distribution by Label")
    
    
    plot_pair_gene_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[6])
    # Make axis symmetrical 
    lim = np.max(np.abs(ax[6].get_xlim()))
    ax[6].set_xlim(-lim, lim)
    ax[6].set_ylim(-lim, lim)
    
    return f
    

import pandas as pd
from matplotlib.axes import Axes

def plot_pair_gene_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.uns["D"][:, cmp1 - 1]], [X.uns["D"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax, color="k")
    ax.set(title="LR Factors")



def add_obs_cmp_both_label(
    X: anndata.AnnData, cmp1: int, cmp2: int, pos1=True, pos2=True, top_perc=1, type="sender"
):
    """Adds if cells in top/bot percentage"""
    if type == "sender":
        factor_type = X.obsm["sc_B"]
    elif type == "receiver":
        factor_type = X.obsm["rc_C"]
  
    pos_neg = [pos1, pos2]

    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold1[cmp - 1]

            else:
                thres_value = top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold1[cmp - 1]

        if i == 1:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold2[cmp - 1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold2[cmp - 1]

        X.obs[f"Cmp{cmp}"] = idx

    if pos1 is True and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is True and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )

    X.obs["Both"] = idx

    return X





def add_obs_cmp_unique_two(X: anndata.AnnData, cmp1: str, cmp2: str):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False), "Label"] = "NoLabel"
    
    return X
