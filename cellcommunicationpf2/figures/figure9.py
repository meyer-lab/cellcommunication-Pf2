"""
Figure 9: Top 15 Weighted LR Pairs by Component

This figure shows bar plots of the top 15 weighted ligand-receptor 
pairs (by absolute weight) for each component from the CC-PF2 decomposition.
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
    """Generate Figure 9 showing top 15 weighted LR pairs per component by absolute weight."""
    
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
    
    # Set up subplots - single row with n_components columns
    n_components = lr_factor.shape[1]
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 10), constrained_layout=True)
    
    # Handle case where n_components = 1
    if n_components == 1:
        axes = [axes]
    
    subplotLabel(axes)

    # Plot top 15 LR pairs by absolute weight for each component
    for comp in range(n_components):
        # Get weights for this component
        weights = lr_factor[:, comp]
        
        # Get indices for top 15 by absolute weight
        top_15_idx = np.argsort(np.abs(weights))[-15:][::-1]  # Descending order by absolute value
        
        # Get the actual weights and labels for top 15
        top_weights = weights[top_15_idx]
        top_labels = [lr_labels.iloc[i] for i in top_15_idx]
        
        # Color bars based on positive/negative weights
        colors = ['red' if w >= 0 else 'blue' for w in top_weights]
        
        ax = axes[comp]
        bars = ax.barh(range(15), top_weights, color=colors, alpha=0.7)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_labels, fontsize=12)  # Increased from 8
        ax.set_xlabel('Weight', fontsize=14)  # Increased from 10
        ax.set_title(f'Component {comp+1}: Top 15 LR Pairs\n(by absolute weight)', fontsize=16)  # Increased from 12
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add weight values on bars
        for i, (bar, weight) in enumerate(zip(bars, top_weights)):
            # Position text based on bar direction
            if weight >= 0:
                x_pos = weight + 0.01 * np.max(np.abs(top_weights))
                ha = 'left'
            else:
                x_pos = weight - 0.01 * np.max(np.abs(top_weights))
                ha = 'right'
            
            ax.text(x_pos, i, f'{weight:.3f}', va='center', ha=ha, fontsize=10)  # Increased from 7
        
        # Invert y-axis so highest absolute weights are at the top
        ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive weight'),
        Patch(facecolor='blue', alpha=0.7, label='Negative weight')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 0.02), fontsize=14)  # Increased from 12

    # Add overall figure title
    plt.suptitle(
        f"Top 15 Weighted LR Pairs per Component by Absolute Weight (R²X = {r2x:.4f})",
        fontsize=20, y=0.98  # Increased from 16
    )

    print("Figure generation complete.")
    return fig