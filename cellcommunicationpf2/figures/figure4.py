"""
Figure 4: R2X Optimization Across Ranks

This figure investigates how the R2X value changes as we vary both the
rise rank (PARAFAC2 rank) and CP rank parameters in the CC-PF2 decomposition.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import getSetup, subplotLabel
from ..utils import run_cc_pf2_workflow


def makeFigure():
    """Generate Figure 4, showing R2X values across different rank combinations."""
    ax, f = getSetup((8, 6), (1, 1))  # Only one plot now
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing data...")
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Define parameter ranges to test
    rise_ranks = [5, 10, 15, 20]  # PARAFAC2 ranks to test
    cp_ranks = [1, 5, 10, 15, 20]  # CP ranks to test

    print(f"Testing {len(rise_ranks)} rise ranks and {len(cp_ranks)} CP ranks...")
    print(f"Rise ranks: {rise_ranks}")
    print(f"CP ranks: {cp_ranks}")

    # Initialize results storage
    results = []

    # Parameters for decomposition
    n_iter_max = 100
    tol = 1e-3
    random_state = 42

    # Grid search over rank combinations
    total_combinations = len(rise_ranks) * len(cp_ranks)
    current_combination = 0

    for rise_rank in rise_ranks:
        for cp_rank in cp_ranks:
            current_combination += 1
            print(
                f"Progress: {current_combination}/{total_combinations} - "
                f"Testing rise_rank={rise_rank}, cp_rank={cp_rank}..."
            )

            try:
                # Run CC-PF2 with current parameters
                adata_copy = adata.copy()  # Create a copy to avoid conflicts
                _, r2x = run_cc_pf2_workflow(
                    adata_copy,
                    rise_rank=rise_rank,
                    lr_pairs=lr_pairs,
                    cp_rank=cp_rank,
                    n_iter_max=n_iter_max,
                    tol=tol,
                    random_state=random_state,
                )

                print(f"  R2X = {r2x:.4f}")

                # Store results
                results.append({"rise_rank": rise_rank, "cp_rank": cp_rank, "r2x": r2x})

            except Exception as e:
                print(f"  Error: {e}")
                # Store NaN for failed runs
                results.append(
                    {"rise_rank": rise_rank, "cp_rank": cp_rank, "r2x": np.nan}
                )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot: Line plot showing trends for each rise rank
    for rise_rank in rise_ranks:
        subset = results_df[results_df["rise_rank"] == rise_rank]
        ax[0].plot(
            subset["cp_rank"],
            subset["r2x"],
            marker="o",
            label=f"Rise Rank {rise_rank}",
            linewidth=2,
            markersize=6,
        )

    ax[0].set_xlabel("CP Rank")
    ax[0].set_ylabel("R²X Value")
    ax[0].set_title("R²X Trends by CP Rank")
    ax[0].legend(title="Rise Rank", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[0].grid(True, alpha=0.3)

    # Add overall figure title
    plt.suptitle("CC-PF2 R²X Optimization Study", fontsize=16, y=0.98)

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean R²X: {results_df['r2x'].mean():.4f}")
    print(f"Std R²X: {results_df['r2x'].std():.4f}")
    print(f"Min R²X: {results_df['r2x'].min():.4f}")
    print(f"Max R²X: {results_df['r2x'].max():.4f}")

    print("Figure generation complete.")
    return f
