"""
Figure 3: Rank-1 Component Weight vs. Sample Size

This figure investigates the behavior of a rank-1 CC-PF2 model by plotting
the component weight for each sample against the number of cells in that sample.
"""

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from ..import_data import (
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import getSetup, subplotLabel
from ..utils import run_cc_pf2_workflow


def makeFigure():
    """Generate Figure 3, showing the correlation between component weight and cell count."""
    ax, f = getSetup((8, 6), (1, 1))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()
    adata_filtered, lr_pairs_filtered = anndata_lrp_overlap(adata, lr_pairs)

    condition_column = "sample"

    # Run a Rank-1 CC-PF2 Model
    rank = 1
    print(f"Running CC-PF2 with rank={rank}...")
    adata_filtered, r2x = run_cc_pf2_workflow(
        adata_filtered,
        rank=rank,
        lr_pairs=lr_pairs_filtered,
        n_iter_max=100,
        tol=1e-3,
        random_state=42,
    )
    print(f"CC-PF2 decomposition R2X: {r2x:.4f}")

    # Prepare Data for Plotting
    # Get the condition factor (component weights per sample)
    condition_factor = adata_filtered.uns["Pf2_A"].flatten()

    # Get the number of cells for each sample
    cell_counts = adata_filtered.obs[condition_column].value_counts()

    # Create mapping from condition_index to sample name
    condition_to_sample = dict(enumerate(adata_filtered.obs[condition_column].cat.categories))

    samples = [condition_to_sample[idx] for idx in range(len(condition_factor))]
    plot_df = pd.DataFrame({
        'Sample': samples,
        'Component Weight': condition_factor,
        'Cell Count': [cell_counts[sample] for sample in samples]
    })

    # Generate the Plot
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