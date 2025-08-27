"""
Figure 2: Stability Analysis for CC-PF2

This figure demonstrates the stability of CC-PF2 decompositions across
data subsampling and different rank selections.
"""

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes
from ..utils import resample, calculate_fms

from ..cc_pf2 import cc_pf2, standardize_cc_pf2
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import getSetup, subplotLabel


def makeFigure():
    """Generate Figure 2 showing stability analyses for CC-PF2."""
    ax, f = getSetup((6, 4), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing data for Figure 2...")
    X = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Add condition indices using sample as the condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    # Parameters for stability plots
    percentList = np.arange(0.0, 25.0, 5.0)
    ranks = [3, 5, 7, 9]
    runs = 3

    # Generate plots
    print("Plotting FMS vs. data dropout...")
    plot_fms_percent_drop(
        X_filtered, ax[0], percentList=percentList, runs=runs, rank=10
    )
    ax[0].set_title("Robustness to Data Subsampling")

    print("Plotting FMS vs. rank...")
    plot_fms_diff_ranks(X_filtered, ax[1], ranksList=ranks, runs=runs)
    ax[1].set_title("Stability Across Ranks")

    return f


def run_cc_pf2_analysis(
    adata: anndata.AnnData, rank: int, random_state: int = 42
) -> anndata.AnnData:
    """Run CC-PF2 decomposition and store results in the AnnData object."""
    adata = adata.copy()
    results, r2x, lr_pairs_filtered = cc_pf2(
        adata, rank, 100, 1e-3, random_state=random_state, use_cache=False
    )
    cp_results, projections = results
    cp_weights, factors = cp_results

    _, factors = standardize_cc_pf2(factors, weights=cp_weights)

    adata.uns["Pf2_A"] = factors[0]
    adata.uns["Pf2_B"] = factors[1]
    adata.uns["Pf2_C"] = factors[2]
    adata.uns["Pf2_D"] = factors[3]
    adata.uns["Pf2_weights"] = cp_weights
    return adata


def plot_fms_percent_drop(
    X: anndata.AnnData,
    ax: Axes,
    percentList: np.ndarray,
    runs: int,
    rank: int = 10,
):
    """Plot Factor Match Score when progressively removing data.

    Creates a reference decomposition and compares it with decompositions
    of increasingly subsampled data.
    """
    # Generate reference decomposition once
    dataX = run_cc_pf2_analysis(X, rank=rank, random_state=42)

    data_list = []
    for j in range(runs):
        # Add baseline case (0% dropout) with perfect FMS score
        data_list.append({"Run": j, "Percentage of Data Dropped": 0.0, "FMS": 1.0})

        for percent in percentList[1:]:
            # Create deterministic but unique seeds for reproducibility
            unique_seed = j * 100 + int(percent)

            # Subsample data based on the percent to drop
            sampled_data: anndata.AnnData = sc.pp.subsample(
                X, fraction=1 - (percent / 100), random_state=unique_seed, copy=True
            )  # type: ignore

            sampled_data = add_cond_idxs(sampled_data, "sample")
            sampledX = run_cc_pf2_analysis(
                sampled_data, rank=rank, random_state=unique_seed + 1
            )

            fmsScore = calculate_fms(dataX, sampledX)
            data_list.append(
                {
                    "Run": j,
                    "Percentage of Data Dropped": percent,
                    "FMS": fmsScore,
                }
            )

    df = pd.DataFrame(data_list)
    sns.lineplot(data=df, x="Percentage of Data Dropped", y="FMS", ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(np.min(percentList), np.max(percentList))


def plot_fms_diff_ranks(
    X: anndata.AnnData,
    ax: Axes,
    ranksList: list[int],
    runs: int,
):
    """Plot Factor Match Score across different rank values.

    For each rank, creates a reference decomposition and compares it with
    decompositions of bootstrapped data samples.
    """
    data_list = []
    base_seed = 42  # Base seed for reproducibility

    for i in ranksList:
        # Calculate reference decomposition once per rank
        dataX = run_cc_pf2_analysis(X, rank=i, random_state=base_seed)

        for j in range(runs):
            # Use deterministic seeds for each run
            run_seed = base_seed + j

            # Create bootstrapped sample with consistent stratification
            resampled_data = resample(X, random_seed=run_seed)
            resampled_data = add_cond_idxs(resampled_data, "sample")
            sampledX = run_cc_pf2_analysis(
                resampled_data, rank=i, random_state=run_seed
            )

            fmsScore = calculate_fms(dataX, sampledX)
            data_list.append({"Run": j, "Rank": i, "FMS": fmsScore})

    df = pd.DataFrame(data_list)
    sns.lineplot(data=df, x="Rank", y="FMS", ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ranksList)
