"""
This file contains functions that are used in multiple figures.
"""

import sys
import time

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from string import ascii_letters
from ..tensor import reorder_table

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

    exec(f"from pf2.figures.{nameOut} import makeFigure", globals())
    ff = makeFigure()

    if ff is not None:
        ff.savefig(
            f"./output/{nameOut}.svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def ds_show(result: tf.Image, ax: plt.Axes):
    """Show datashader results."""
    result = tf.dynspread(result, threshold=0.95, max_px=5)
    result = tf.set_background(result, "white")
    img_rev = result.data[::-1]
    mpl_img = np.dstack(
        [
            img_rev & 0x0000FF,
            (img_rev & 0x00FF00) >> 8,
            (img_rev & 0xFF0000) >> 16,
        ]
    )

    ax.imshow(mpl_img)


def get_canvas(points: np.ndarray) -> ds.Canvas:
    """Compute bounds on a space with appropriate padding"""
    min_xy = np.nanmin(points, axis=0)
    assert min_xy.size == 2
    max_xy = np.nanmax(points, axis=0)

    mins = np.round(min_xy - 0.05 * (max_xy - min_xy))
    maxs = np.round(max_xy + 0.05 * (max_xy - min_xy))

    canvas = ds.Canvas(
        plot_width=900,
        plot_height=900,
        x_range=(mins[0], maxs[0]),
        y_range=(mins[1], maxs[1]),
    )

    return canvas


  