"""
Figure S4a_c: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils import expression_product_matrix

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")

    # Create barplot of percent of cell-cell interactions that are non-zero for each sample
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]

    pairs = [["PTN", "PTPRZ1"], ["PTN", "SDC1"], ["COL4A5", "SDC1"]]

    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    
    for i, (lig, rec) in enumerate(pairs):
        results_data = []
        pair_name = f"{lig}-{rec}"
        
        # Calculate percentage of non-zero interactions for each sample
        for sample in X.obs["sample"].unique():
            # Get condition for this sample
            sample_condition = X.obs[X.obs["sample"] == sample]["condition"].iloc[0]
            
            X_mdc_sender_sample = X_mdc_sender[X_mdc_sender.obs["sample"] == sample]
            X_mdc_receiver_sample = X_mdc_receiver[X_mdc_receiver.obs["sample"] == sample]
            
            if len(X_mdc_sender_sample) > 0 and len(X_mdc_receiver_sample) > 0:
                df = expression_product_matrix(X_mdc_sender_sample, X_mdc_receiver_sample, lig, rec)
                
                # Count total interactions and non-zero interactions
                total_interactions = df.size  # Total number of cell-cell pairs
                nonzero_interactions = np.sum(df.values > 0)  # Number of non-zero interactions
                
                if total_interactions > 0:
                    percent_nonzero = (nonzero_interactions / total_interactions) * 100
                    results_data.append({
                        'pair': pair_name,
                        'sample': sample,
                        'condition': sample_condition,
                        'percent_nonzero': percent_nonzero,
                        'nonzero_interactions': nonzero_interactions,
                        'total_interactions': total_interactions
                    })
    
        # Convert to DataFrame and create boxplot
        results_df = pd.DataFrame(results_data)
        print("Percentage of non-zero interactions within each sample:")
        print(results_df)

        # Create barplot showing distribution of percentages across samples
        
        sns.barplot(data=results_df, x='sample', y='percent_nonzero', hue='condition', ax=ax[i], palette=pal)
        ax[i].set_xlabel(f'Ligand-Receptor Pair: {pair_name}')
        ax[i].set_ylabel('% Non-Zero Interactions')
        ax[i].tick_params(axis='x', rotation=45)


    return f