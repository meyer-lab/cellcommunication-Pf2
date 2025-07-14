"""
This file contains functions that are used in multiple figures.
"""

import sys
import time
from string import ascii_letters

import matplotlib
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ..cc_pf2 import cc_pf2, standardize_cc_pf2
import anndata
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_CMAP = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
DIVERGING_CMAP = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
LIGHT_DIVERGING = sns.diverging_palette(240, 10, as_cmap=True)


def getSetup(
    figsize: tuple[int, int], gridd: tuple[int, int]
) -> tuple[list[plt.Axes], Figure]:
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, layout="constrained")
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return ax, f


def subplotLabel(axs: list[plt.Axes]):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec(f"from cellcommunicationpf2.figures.{nameOut} import makeFigure", globals())
    ff = makeFigure()

    if ff is not None:
        ff.savefig(
            f"./output/{nameOut}.svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]

    counts = np.zeros((np.amax(sgIndex.to_numpy()) + 1, 1))

    cond_mean = np.linalg.norm(X.uns["Pf2_A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct


def run_cc_pf2_workflow(
    adata: anndata.AnnData,
    rank: int,
    lr_pairs: pd.DataFrame,
    cp_rank: int | None = None,
    n_iter_max: int = 100,
    tol: float = 1e-3,
    random_state: int | None = None,
) -> tuple[anndata.AnnData, float]:
    """
    Executes the complete CC-PF2 workflow: decomposition, standardization,
    condition factor correction, and result storage.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with expression data.
    rank : int
        The rank for the decomposition.
    lr_pairs : pd.DataFrame
        The ligand-receptor pairs used in the decomposition.
    cp_rank : int, optional
        The rank for the final CP decomposition. If None, defaults to `rank`.
    n_iter_max : int
        Maximum number of iterations for the decomposition.
    tol : float
        Convergence tolerance for the decomposition.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[anndata.AnnData, float]
        A tuple containing the updated AnnData object with stored results
        and the R2X value of the decomposition.
    """

    # 1. Run the CC-PF2 decomposition
    results, r2x = cc_pf2(
        adata,
        rank,
        n_iter_max=n_iter_max,
        tol=tol,
        cp_rank=cp_rank,
        random_state=random_state,
    )
    (cp_weights, factors), projections = results

    # 2. Standardize the factors for interpretability
    _, factors, projections = standardize_cc_pf2(
        factors, projections, weights=cp_weights
    )

    # 3. Correct the condition factors
    # Temporarily store the condition factor to be used by the correction function
    adata.uns["Pf2_A"] = factors[0]
    # Run the correction
    corrected_factor_A = correct_conditions(adata)
    # Replace the original factor with the corrected one
    factors[0] = corrected_factor_A

    # Store factors in AnnData object for easy access by plotting functions
    adata.uns["Pf2_A"] = factors[0]  # Condition factor
    adata.uns["Pf2_B"] = factors[1]  # Sender cells factor
    adata.uns["Pf2_C"] = factors[2]  # Receiver cells factor
    adata.uns["Pf2_D"] = factors[3]  # LR pairs factor

    # Store the LR pairs used in the decomposition, as this is not handled by the workflow
    adata.uns["Pf2_lr_pairs"] = lr_pairs.reset_index(drop=True)

    return adata, r2x
