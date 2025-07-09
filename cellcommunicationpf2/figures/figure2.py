"""
Figure S4
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
    """Generate Figure S4 showing stability analyses for CC-PF2."""
    ax, f = getSetup((10, 4), (1, 2))
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
    """Helper function to run and store CC-PF2 results."""
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
    """Calculates FMS between 2 factors, accounting for the 4 factors from cc_pf2."""
    # Factors are now A, B, C, D from .uns
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))

    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]]
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    # Skip sender/receiver cell factors (modes 1 and 2) as they are most variable
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=[1, 2])


def plot_fms_percent_drop(
    X: anndata.AnnData,
    ax: Axes,
    percentList: np.ndarray,
    runs: int,
    rank: int = 10,
):
    """Plots FMS score when percentage is removed from data."""
    # Use the new helper function for the reference decomposition
    dataX = run_cc_pf2_analysis(X, rank=rank, random_state=42)

    data_list = []
    for j in range(runs):
        # Add the score for 0% dropout (perfect match with itself)
        data_list.append({"Run": j, "Percentage of Data Dropped": 0.0, "FMS": 1.0})

        for percent in percentList[1:]:
            # Create a unique random state for this specific run and percentage
            unique_seed = j * 100 + int(percent)
            
            sampled_data: anndata.AnnData = sc.pp.subsample(
                X, fraction=1 - (percent / 100), random_state=unique_seed, copy=True
            )  # type: ignore
            # Must re-add condition indices after subsampling for cc_pf2 to work
            sampled_data = add_cond_idxs(sampled_data, "sample")
            # Use a different unique seed for the decomposition
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


def resample(data: anndata.AnnData) -> anndata.AnnData:
    """Bootstrapping dataset by stratifying by sample."""
    resampled_indices = []
    for sample in data.obs["sample"].unique():
        # Get the original indices for all cells in the current sample
        sample_cell_indices = np.where(data.obs["sample"] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Generate random indices *within* this sample's cell list.
        # This uses the same np.random.randint methodology as before.
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)

        # Select the original cell indices based on the random local indices
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData object from the stratified, resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


def plot_fms_diff_ranks(
    X: anndata.AnnData,
    ax: Axes,
    ranksList: list[int],
    runs: int,
):
    """Plots FMS when using different Pf2 components"""
    data_list = []

    for i in ranksList:
        # Calculate the reference decomposition ONCE for this rank.
        dataX = run_cc_pf2_analysis(X, rank=i, random_state=42)
        for j in range(0, runs, 1):
            # Compare each run against the single reference decomposition.
            resampled_data = resample(X)
            # Must re-add condition indices after resampling
            resampled_data = add_cond_idxs(resampled_data, "sample")
            sampledX = run_cc_pf2_analysis(resampled_data, rank=i, random_state=j)

            fmsScore = calculateFMS(dataX, sampledX)
            data_list.append({"Run": j, "Rank": i, "FMS": fmsScore})

    df = pd.DataFrame(data_list)
    sns.lineplot(data=df, x="Rank", y="FMS", ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ranksList)
