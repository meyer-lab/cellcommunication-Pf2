"""
Figure A1: FMS across RISE ranks (only) for COVID-19
"""

import anndata
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from ..utils import resample
from parafac2.parafac2 import parafac2_nd, store_pf2
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
)
from .common import getSetup, subplotLabel
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_balf_covid()

    # Add condition indices using sample as the condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    # Parameters for stability plots
    ranks = list(range(1, 81, 5))
    ranks = list(range(1, 11, 5))
    runs = 3

    print("Plotting FMS vs. rank...")
    plot_fms_r2x_diff_ranks(X_filtered, ax[0], ax[1], ranksList=ranks, runs=runs)
    ax[0].set_title(f"RISE on COVID-19 scRNA-seq: {X_filtered.shape[1]} genes")
    ax[1].set_title(f"RISE on COVID-19 scRNA-seq: {X_filtered.shape[1]} genes")

    return f


def calculateFMS(A: anndata.AnnData, B: anndata.AnnData):
    """Calculates FMS between 2 factors"""
    factors = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    A_CP = CPTensor(
        (
            A.uns["Pf2_weights"],
            factors,
        )
    )

    factors = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    B_CP = CPTensor(
        (
            B.uns["Pf2_weights"],
            factors,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=1)  # type: ignore


def plot_fms_r2x_diff_ranks(
    X: anndata.AnnData,
    ax1: Axes,
    ax2: Axes,
    ranksList: list[int],
    runs: int,
):
    """Plots FMS when using different RISE components"""
    fmsLists = []
    r2xLists = []

    for j in range(0, runs, 1):
        scores = []
        r2x_scores = []
        for i in ranksList:
            dataX, r2x = rise_store_r2x(X, rank=i, n_iter_max=1000, tolerance=1e-9)
            sampledX, _ = rise_store_r2x(resample(X), rank=i, n_iter_max=1000, tolerance=1e-9)
            fmsScore = calculateFMS(dataX, sampledX)
            scores.append(fmsScore)
            # Calculate R2X for X only (not resampled)
            r2x_scores.append(r2x)
        fmsLists.append(scores)
        r2xLists.append(r2x_scores)

    runsList_df = []
    for i in range(0, runs):
        for _j in range(0, len(ranksList)):
            runsList_df.append(i)
    ranksList_df = []
    for _i in range(0, runs):
        for j in range(0, len(ranksList)):
            ranksList_df.append(ranksList[j])
    fmsList_df = []
    for sublist in fmsLists:
        fmsList_df += sublist
    r2xList_df = []
    for sublist in r2xLists:
        r2xList_df += sublist
        
        
    df = pd.DataFrame(
        {"Run": runsList_df, "Component": ranksList_df, "FMS": fmsList_df, "R2X": r2xList_df}
    )

    # Plot FMS
    sns.lineplot(data=df, x="Component", y="FMS", ax=ax1, label="FMS")
    ax1.set_ylim(0, 1)

    # Plot R2X on a secondary y-axis

    sns.lineplot(data=df, x="Component", y="R2X", ax=ax2, color="orange", label="R2X")
    ax2.set_ylim(0, np.max(df["R2X"]) + 0.02)


def rise_store_r2x(X: anndata.AnnData, rank: int, n_iter_max: int, tolerance: float, random_state: int = None):
    """Runs RISE and stores the results."""
    pf2_out, r2x = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=n_iter_max
    )
    X = store_pf2(X, pf2_out)
    
    return X, r2x