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
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms

from ..cc_pf2 import cc_pf2, standardize_cc_pf2
from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
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
    X_filtered, _ = anndata_lrp_overlap(X, lr_pairs)

    # Add condition indices using sample as the condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X_filtered, condition_column)

    # Parameters for stability plots
    percentList = np.arange(0.0, 25.0, 5.0)
    ranks = [3, 5, 7, 9]
    runs = 3

    # Generate plots
    print("Plotting FMS vs. data dropout...")
    plot_fms_percent_drop(X_filtered, ax[0], percentList=percentList, runs=runs, rank=10)
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
    results, r2x = cc_pf2(adata, rank, 100, 1e-3, random_state=random_state)
    cp_results, projections = results
    cp_weights, factors = cp_results

    _, factors, _ = standardize_cc_pf2(factors, projections, weights=cp_weights)

    adata.uns["Pf2_A"] = factors[0]
    adata.uns["Pf2_B"] = factors[1]
    adata.uns["Pf2_C"] = factors[2]
    adata.uns["Pf2_D"] = factors[3]
    adata.uns["Pf2_weights"] = cp_weights
    return adata


def calculateFMS(A: anndata.AnnData, B: anndata.AnnData) -> float:
    """Calculate FMS between two CC-PF2 decompositions stored in AnnData objects.
    
    Skips comparison of sender/receiver factors (modes 1 and 2) as they are most variable.
    """
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))

    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]]
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    return fms(A_CP, B_CP, consider_weights=False, skip_mode=[1, 2])


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
            sampledX = run_cc_pf2_analysis(sampled_data, rank=rank, random_state=unique_seed + 1)

            fmsScore = calculateFMS(dataX, sampledX)
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


def resample(data: anndata.AnnData, random_seed: int = None) -> anndata.AnnData:
    """Perform stratified bootstrap sampling by resampling cells within each sample.
    
    This maintains the same number of cells per sample in the resampled dataset.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    resampled_indices = []
    for sample in data.obs["sample"].unique():
        # Get indices of cells in this sample
        sample_cell_indices = np.where(data.obs["sample"] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Sample with replacement within this sample's cells
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData from resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


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
            sampledX = run_cc_pf2_analysis(resampled_data, rank=i, random_state=run_seed)

            fmsScore = calculateFMS(dataX, sampledX)
            data_list.append({"Run": j, "Rank": i, "FMS": fmsScore})

    df = pd.DataFrame(data_list)
    sns.lineplot(data=df, x="Rank", y="FMS", ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ranksList)
