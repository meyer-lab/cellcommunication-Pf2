"""
Figure 9: Top and Bottom Weighted LR Pairs by Component

This figure shows bar plots of the top 5 and bottom 5 weighted ligand-receptor 
pairs for each component from the CC-PF2 decomposition.
"""

import numpy as np
from matplotlib import pyplot as plt

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import subplotLabel
from ..utils import run_cc_pf2_workflow


def makeFigure():
    """Generate Figure 9 showing top and bottom weighted LR pairs per component."""
    
    # Import and prepare data
    print("Importing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata, condition_column)

    # Parameters for CC-PF2
    rise_rank = 30
    cp_rank = 10
    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    # Run CC-PF2
    print(f"Running CC-PF2 with rank={rise_rank} and cp_rank={cp_rank}...")
    adata_filtered, r2x = run_cc_pf2_workflow(
        adata_filtered,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )

    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")

    # Get LR factor and pairs
    lr_factor = adata_filtered.uns["Pf2_D"]  # Shape: (n_lr_pairs, n_components)
    lr_pairs_used = adata_filtered.uns["Pf2_lr_pairs"]
    
    # Create LR pair labels
    lr_labels = lr_pairs_used["ligand"] + " → " + lr_pairs_used["receptor"]
    
    # Set up subplots - 2 rows (top/bottom) x n_components columns
    n_components = lr_factor.shape[1]
    fig, axes = plt.subplots(2, n_components, figsize=(4*n_components, 8), constrained_layout=True)
    
    # Handle case where n_components = 1
    if n_components == 1:
        axes = axes.reshape(2, 1)
    
    # Flatten for easy iteration
    ax_flat = axes.flatten()
    subplotLabel(ax_flat)

    # Plot top and bottom 5 LR pairs for each component
    for comp in range(n_components):
        # Get weights for this component
        weights = lr_factor[:, comp]
        
        # Get indices for top and bottom 5
        top_5_idx = np.argsort(weights)[-5:][::-1]  # Descending order
        bottom_5_idx = np.argsort(weights)[:5]  # Ascending order
        
        # Top 5 plot (first row)
        ax_top = axes[0, comp]
        top_weights = weights[top_5_idx]
        top_labels = [lr_labels.iloc[i] for i in top_5_idx]
        
        bars_top = ax_top.barh(range(5), top_weights, color='red', alpha=0.7)
        ax_top.set_yticks(range(5))
        ax_top.set_yticklabels(top_labels, fontsize=8)
        ax_top.set_xlabel('Weight')
        ax_top.set_title(f'Component {comp+1}: Top 5 LR Pairs', fontsize=10)
        ax_top.grid(True, alpha=0.3)
        
        # Add weight values on bars
        for i, (bar, weight) in enumerate(zip(bars_top, top_weights)):
            ax_top.text(weight + 0.01*np.max(np.abs(top_weights)), i, 
                       f'{weight:.3f}', va='center', fontsize=8)
        
        # Bottom 5 plot (second row)  
        ax_bottom = axes[1, comp]
        bottom_weights = weights[bottom_5_idx]
        bottom_labels = [lr_labels.iloc[i] for i in bottom_5_idx]
        
        bars_bottom = ax_bottom.barh(range(5), bottom_weights, color='blue', alpha=0.7)
        ax_bottom.set_yticks(range(5))
        ax_bottom.set_yticklabels(bottom_labels, fontsize=8)
        ax_bottom.set_xlabel('Weight')
        ax_bottom.set_title(f'Component {comp+1}: Bottom 5 LR Pairs', fontsize=10)
        ax_bottom.grid(True, alpha=0.3)
        
        # Add weight values on bars
        for i, (bar, weight) in enumerate(zip(bars_bottom, bottom_weights)):
            x_pos = weight - 0.01*np.max(np.abs(bottom_weights)) if weight < 0 else weight + 0.01*np.max(np.abs(bottom_weights))
            ax_bottom.text(x_pos, i, f'{weight:.3f}', va='center', fontsize=8)

    # Add overall figure title
    plt.suptitle(
        f"Top and Bottom 5 Weighted LR Pairs per Component (R²X = {r2x:.4f})",
        fontsize=16, y=0.98
    )

    print("Figure generation complete.")
    return fig