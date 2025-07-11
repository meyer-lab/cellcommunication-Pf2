"""
Figure 3: Rank-1 Component Weight vs. Sample Size

This figure investigates the behavior of a rank-1 CC-PF2 model by plotting
the component weight for each sample against the number of cells in that sample.
"""

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from ..cc_pf2 import cc_pf2, standardize_cc_pf2
from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import getSetup, subplotLabel


def makeFigure():
    """Generate Figure 3, showing the correlation between component weight and cell count."""
    ax, f = getSetup((8, 6), (1, 1))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()
    adata_filtered, _ = anndata_lrp_overlap(adata, lr_pairs)

    # Add condition indices using patient sample as the condition
    condition_column = "sample"
    adata_filtered = add_cond_idxs(adata_filtered, condition_column)

    # --- Run a Rank-1 CC-PF2 Model ---
    rank = 1
    print(f"Running CC-PF2 with rank={rank}...")
    results, r2x = cc_pf2(
        adata_filtered, rank, n_iter_max=100, tol=1e-3, random_state=42
    )
    cp_results, projections = results
    cp_weights, factors = cp_results

    # Standardize factors
    _, factors, _ = standardize_cc_pf2(factors, projections, weights=cp_weights)
    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")

    # --- Prepare Data for Plotting ---
    # Get the condition factor (component weights per sample)
    condition_factor = factors[0].flatten()  # First factor (A) contains condition weights
    
    # Get the number of cells for each sample
    cell_counts = adata_filtered.obs[condition_column].value_counts()
    
    # Create mapping from condition_index to sample name
    condition_to_sample = {}
    for idx, sample in enumerate(adata_filtered.obs[condition_column].cat.categories):
        condition_to_sample[idx] = sample
    
    # Verify the mapping is complete
    print(f"Mapping condition indices to samples: {condition_to_sample}")
    
    # Create DataFrame with explicit mapping
    plot_data = []
    for idx, weight in enumerate(condition_factor):
        sample = condition_to_sample[idx]
        cell_count = cell_counts[sample]
        plot_data.append({
            'Sample': sample,
            'Component Weight': weight,
            'Cell Count': cell_count
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # --- Generate the Plot ---
    print("Generating plot...")
    sns.regplot(
        data=plot_df, x="Cell Count", y="Component Weight", ax=ax[0], ci=None
    )

    # Calculate and display Pearson correlation
    r, p = pearsonr(plot_df["Cell Count"], plot_df["Component Weight"])
    ax[0].text(
        0.05,
        0.95,
        f"Pearson r = {r:.3f}\np-value = {p:.3e}",
        transform=ax[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )

    # Label axes
    ax[0].set_xlabel("Cell Count")
    ax[0].set_ylabel("Component Weight")

    return f