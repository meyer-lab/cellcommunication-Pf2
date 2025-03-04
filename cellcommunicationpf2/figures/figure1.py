"""
Figure 1: XX
"""

from .common import getSetup, subplotLabel
from ..import_data import balf_covid

def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)
    
    # X = balf_covid()
    # print(X)
    
    

    return f
