"""
Figure 1: CC-PF2 Factor Visualization

This figure shows the factors from a rank-10 CC-PF2 decomposition
of the BALF COVID-19 dataset.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..cc_pf2 import cc_pf2, standardize_cc_pf2
from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import subplotLabel
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)


def makeFigure():
    """Generate Figure 1 showing CC-PF2 factor heatmaps using specialized plotting functions."""
    # Create figure with 4 subplots (2x2 grid) with constrained_layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    ax = axes.flatten()
    subplotLabel(ax)
    
    # Import and prepare data
    print("Importing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()
    
    # Filter data to include only genes in the ligand-receptor pairs
    print("Filtering data...")
    adata_filtered, lr_pairs_filtered = anndata_lrp_overlap(adata, lr_pairs)
    
    # Add condition indices
    condition_column = "condition"
    adata_filtered = add_cond_idxs(adata_filtered, condition_column)
    
    # Parameters for CC-PF2
    rank = 6
    n_iter_max = 100
    tol = 1e-3
    random_state = 42
    
    # Run CC-PF2
    print(f"Running CC-PF2 with rank={rank}...")
    results, r2x = cc_pf2(
        adata_filtered, rank, n_iter_max, tol, random_state=random_state
    )
    cp_results, projections = results
    cp_weights, factors = cp_results
    
    # Clean non-finite values before standardization to prevent errors
    for i in range(len(factors)):
        if not np.all(np.isfinite(factors[i])):
            factors[i] = np.nan_to_num(factors[i], nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize factors for better interpretability
    print("Standardizing factors...")
    _, factors, projections = standardize_cc_pf2(factors, projections, weights=cp_weights)
    
    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")
    
    # Store factors in AnnData object for plotting functions
    adata_filtered.uns["Pf2_A"] = factors[0]  # Condition factor
    adata_filtered.uns["Pf2_B"] = factors[1]  # Sender cells factor
    adata_filtered.uns["Pf2_C"] = factors[2]  # Receiver cells factor
    
    # Map LR pairs factor to genes using dictionary for O(1) lookups
    lr_factor_for_genes = np.zeros((adata_filtered.n_vars, rank))
    gene_to_idx = {gene: i for i, gene in enumerate(adata_filtered.var_names)}
    
    for i, row in lr_pairs_filtered.iterrows():
        ligand_idx = gene_to_idx.get(row['ligand'])
        receptor_idx = gene_to_idx.get(row['receptor'])
        
        if ligand_idx is not None:
            lr_factor_for_genes[ligand_idx] = factors[3][i]
        
        if receptor_idx is not None:
            lr_factor_for_genes[receptor_idx] = factors[3][i]
    
    adata_filtered.varm["Pf2_D"] = np.nan_to_num(lr_factor_for_genes, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate heatmaps for each factor
    print("Generating heatmaps...")
    
    # 1. Condition factor (A)
    plot_condition_factors(adata_filtered, ax[0], cond=condition_column)
    ax[0].set_title("Factor 0: Conditions")
    
    # 2. Sender cells factor (B)
    plot_eigenstate_factors(adata_filtered, ax[1], factor_type="Pf2_B")
    ax[1].set_title("Factor 1: Sender Cell Eigenstates")
    
    # 3. Receiver cells factor (C)
    plot_eigenstate_factors(adata_filtered, ax[2], factor_type="Pf2_C")
    ax[2].set_title("Factor 2: Receiver Cell Eigenstates")
    
    # 4. LR pairs factor (D) 
    plot_lr_factors(adata_filtered, ax[3], trim=True)
    ax[3].set_title("Factor 3: LR Pairs")
    
    # Add overall figure title with R2X information
    plt.suptitle(f"CC-PF2 Decomposition (Rank {rank}, R²X = {r2x:.4f})", 
                fontsize=16, y=0.995)
    
    print("Figure generation complete.")
    return fig
