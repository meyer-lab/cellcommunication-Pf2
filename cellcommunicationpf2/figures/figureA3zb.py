"""
Figure A3zb: XXX
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
)
import scanpy as sc


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 5
    
    X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender_top = add_obs_cmp_label(X_epi_sender, cmp=ccc_rise_cmp, pos=True, top_perc=10, type="sender")
    X_epi_sender_top = add_obs_cmp_unique_one(X_epi_sender_top, cmp=ccc_rise_cmp)
    X_epi_sender_top = X_epi_sender_top[X_epi_sender_top.obs["Label"].isin(["Cmp5"])]
    X_epi_sender_top.obs["Label"] = "Cmp5_Top"
   
    X_epi_sender_bottom = add_obs_cmp_label(X_epi_sender, cmp=ccc_rise_cmp, pos=False, top_perc=10, type="sender")
    X_epi_sender_bottom = add_obs_cmp_unique_one(X_epi_sender_bottom, cmp=ccc_rise_cmp)
    X_epi_sender_bottom = X_epi_sender_bottom[X_epi_sender_bottom.obs["Label"].isin(["Cmp5"])]
    X_epi_sender_bottom.obs["Label"] = "Cmp5_Bottom"
    
    combined_X = anndata.concat([X_epi_sender_top, X_epi_sender_bottom])
    
    method_test = ["t-test", "wilcoxon"]
    axs = 0
    for method in method_test:
        sc.tl.rank_genes_groups(combined_X, "Label", method=method)
        results = combined_X.uns["rank_genes_groups"]
        top_n = 20
        top_idx_top = np.argsort(results['scores']['Cmp5_Top'])[-top_n:][::-1]
        top_genes_top = results['names']['Cmp5_Top'][top_idx_top]
        top_scores_top = results['scores']['Cmp5_Top'][top_idx_top]
        
        sns.barplot(
            x=top_scores_top,
            y=top_genes_top,
            ax=ax[axs],
        )
        ax[axs].set_xlabel(method)
        ax[axs].set_ylabel("Gene")
        
        top_idx_bottom = np.argsort(results['scores']['Cmp5_Bottom'])[-top_n:][::-1]
        top_genes_bottom = results['names']['Cmp5_Bottom'][top_idx_bottom]
        top_scores_bottom = results['scores']['Cmp5_Bottom'][top_idx_bottom]
        
        sns.barplot(
            x=top_scores_bottom,
            y=top_genes_bottom,
            ax=ax[axs+2],
        )
        ax[axs+2].set_xlabel(method)
        ax[axs+2].set_ylabel("Gene")
        axs += 1

    ax[0].set_title("Top 10 perc vs Bot 10 perc: t-test")
    ax[1].set_title("Top 10 perc vs Bot 10 perc: Wilcoxon")
    ax[2].set_title("Bottom 10 perc vs Top 10 perc: t-test")
    ax[3].set_title("Bottom 10 perc vs Top 10 perc: Wilcoxon")

        
    return f

