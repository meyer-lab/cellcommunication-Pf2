import numpy as np
import anndata
import pandas as pd
import seaborn as sns
from ...utils import rise_store_r2x, calculate_fms_rise, resample
from matplotlib.axes import Axes

def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)
    
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
            print(f"Run {j+1}, Rank {i}")
            dataX, r2x = rise_store_r2x(X, rank=i, n_iter_max=1000, tolerance=1e-9)
            sampledX, _ = rise_store_r2x(resample(X), rank=i, n_iter_max=1000, tolerance=1e-9)
            fmsScore = calculate_fms_rise(dataX, sampledX)
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
