"""
Figure 3: Rank-1 Component Weight vs. Sample Size

This figure investigates the behavior of a rank-1 CC-PF2 model by plotting
the component weight for each sample against the number of cells in that sample.
"""

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import getSetup, subplotLabel
from ..utils import run_cc_pf2_workflow
from .commonFuncs.plotFactors import plot_condition_factors


def makeFigure():
    """Generate Figure 3, showing the correlation between component weight and cell count."""
    ax, f = getSetup((12, 6), (1, 2))  # Changed to 2 subplots
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    condition_column = "sample"

    # Run a Rank-1 CC-PF2 Model
    
    rise_rank = 30
    cp_rank = 1

    adata_filtered, r2x = run_cc_pf2_workflow(
        adata,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        cp_rank=cp_rank,
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

    # Get sample names in the order of condition_unique_idxs
    idxs = np.argsort(adata_filtered.obs["condition_unique_idxs"].unique())
    samples = pd.Series(adata_filtered.obs[condition_column].unique())[idxs]

    plot_df = pd.DataFrame(
        {
            "Sample": samples,
            "Component Weight": condition_factor,
            "Cell Count": [cell_counts[sample] for sample in samples],
        }
    )

    # Generate the Scatter Plot
    print("Generating scatter plot...")
    sns.regplot(data=plot_df, x="Cell Count", y="Component Weight", ax=ax[0], ci=None)

    # Add sample name labels to each point
    for _, row in plot_df.iterrows():
        ax[0].annotate(
            row["Sample"],
            (row["Cell Count"], row["Component Weight"]),
            xytext=(3, 3),  # offset from point in pixels
            textcoords="offset points",
            fontsize=6,  # tiny text
            alpha=0.7,
            ha="left",
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

    # Label axes for scatter plot
    ax[0].set_xlabel("Cell Count")
    ax[0].set_ylabel("Component Weight")
    ax[0].set_title("Component Weight vs Cell Count")

    # Generate the Condition Factor Heatmap
    print("Generating condition factor heatmap...")
    plot_condition_factors(adata_filtered, ax[1], cond=condition_column)
    ax[1].set_title("Condition Factors")

    return f
