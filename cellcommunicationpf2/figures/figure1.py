"""
Figure 1: XX
"""

from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)

    return f
