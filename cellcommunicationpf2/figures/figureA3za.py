"""
Figure A3za: XXX
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
    pos_perc = [True, False]
    axs = 0
    
    for i in pos_perc:
        X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
        X_epi_sender = add_obs_cmp_label(X_epi_sender, cmp=ccc_rise_cmp, pos=i, top_perc=10, type="sender")
        X_epi_sender = add_obs_cmp_unique_one(X_epi_sender, cmp=ccc_rise_cmp)

        method_test = ["logreg", "wilcoxon"]
        
        for method in method_test:
            sc.tl.rank_genes_groups(X_epi_sender, "Label", method=method)
            results = X_epi_sender.uns["rank_genes_groups"]
            print(results)
            top_n = 20
            top_idx = np.argsort(results['scores']['Cmp5'])[-top_n:][::-1]
            top_genes = results['names']['Cmp5'][top_idx]
            top_scores = results['scores']['Cmp5'][top_idx]
            
            sns.barplot(
                x=top_scores,
                y=top_genes,
                ax=ax[axs],
            )
            ax[axs].set_xlabel(method)
            ax[axs].set_ylabel("Gene")
            axs += 1
    
    ax[0].set_title("Top 10 perc vs all other cells: logreg")
    ax[1].set_title("Top 10 perc vs all other cells: Wilcoxon")
    ax[2].set_title("Bottom 10 perc vs all other cells: logreg")
    ax[3].set_title("Bottom 10 perc vs all other cells: Wilcoxon")
        
        
    return f

