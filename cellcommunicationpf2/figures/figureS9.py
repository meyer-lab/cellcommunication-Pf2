"""
Figure S1e_g: Cell type proportions across conditions in BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
import gseapy


def makeFigure():
    ax, f = getSetup((4, 4), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    
    cmp = 4
    
    # Create a pandas Series with lr_pairs as index and D values as ranking scores
    ranking_data = pd.Series(
        data=X.uns["D"][:, cmp-1],  # Use column indexing, not row indexing
        index=X.uns["lr_pairs"]
    )
    
    # Sort by ranking score (descending order)
    ranking_data = ranking_data.sort_values(ascending=False)
    
    print(f"Ranking data shape: {ranking_data.shape}")
    print(f"Top 5 ranked pairs:")
    print(ranking_data.head())
    
    # Run GSEA prerank
    gsea_results = gseapy.prerank(
        rnk=ranking_data,
        gene_sets=X.uns["lr_pairs"],
        outdir=None,
        no_plot=True,
    )

    return f
