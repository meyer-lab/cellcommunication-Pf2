"""
Figure 1
"""

from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)

    return f
