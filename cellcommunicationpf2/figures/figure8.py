"""
Figure 8: FMS and R2X across data percentage with fixed CPD and RISE ranks for COVID-19
"""

import pandas as pd
import seaborn as sns
import numpy as np
from ..import_data import add_cond_idxs, import_balf_covid, import_ligand_receptor_pairs
from .common import getSetup, subplotLabel
from ..tensor import run_fms_r2x_analysis, calculate_interaction_tensor


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare dat
    print("Importing and preparing")
    X = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    # Run FMS and R2X analysis across data percentages
    percentage_list = list(range(100, 45, -5))
    fixed_rank = 8
    runs = 3
    
    all_results = []
    for percentage in percentage_list:
        print(f"Running analysis for {percentage}% of data")
        
        if percentage < 100:
            n_cells = X_filtered.n_obs
            n_keep = int(n_cells * percentage / 100)
            cells_to_keep = np.random.choice(n_cells, n_keep, replace=False)
            X_subsampled = X_filtered[cells_to_keep]
        else:
            X_subsampled = X_filtered
        
        # Calculate interaction tensor with subsampled data
        interaction_tensor = calculate_interaction_tensor(
            X_subsampled, lr_pairs, rise_rank=35
        )
        
        df = run_fms_r2x_analysis(
            interaction_tensor, rank_list=[fixed_rank], runs=runs, svd_init="random"
        )
        df['Data_Percentage'] = percentage
        all_results.append(df)
    
    df = pd.concat(all_results, ignore_index=True)

    sns.lineplot(data=df, x="Data_Percentage", y="FMS", ax=ax[0])
    ax[0].set_xlabel("Data Percentage (%)")
    ax[0].set_ylabel("FMS")
    ax[0].set_ylim(0, 1)
    ax[0].invert_xaxis()

    sns.lineplot(data=df, x="Data_Percentage", y="R2X", ax=ax[1], color="orange")
    ax[1].set_xlabel("Data Percentage (%)")
    ax[1].set_ylabel("R2X")
    ax[1].set_ylim(0, np.max(df["R2X"]) + 0.02)
    ax[1].invert_xaxis()

    return f