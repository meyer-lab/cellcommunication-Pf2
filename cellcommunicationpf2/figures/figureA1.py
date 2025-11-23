"""
Figure A2b: CCC-RISE on BALF COVID-19 data. Plotting factors for conditions, sender cells, receiver cells, and ligand-receptor pairs.
"""

from .common import (
    subplotLabel,
    getSetup,
)
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..tensor import run_ccc_rise_workflow
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
import anndata
from .commonFuncs.plotGeneral import rotate_yaxis
import numpy as np


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    
    x = np.linspace(-5, 5, 1000)

    # High peak in the middle (narrow normal distribution)
    high_peak = np.exp(-x**2 / (2 * 0.2**2))
    high_peak /= high_peak.sum()  # Normalize

    # Wide distribution (broad normal distribution)
    wide = np.exp(-x**2 / (2 * 2.0**2))
    wide /= wide.sum()  # Normalize


    # Wide distribution with three distinct humps/peaks
    three_hump_peaks = [(-3, 0.7), (0, 0.7), (3, 0.7)]  # (center, width)
    three_hump = sum(np.exp(-(x-c)**2 / (2 * w**2)) for c, w in three_hump_peaks)
    three_hump /= three_hump.sum()  # Normalize

    ax[0].plot(x, high_peak, label='High Peak (Narrow)')
    ax[1].plot(x, wide, label='Wide Distribution')
    ax[2].plot(x, three_hump, label='Three-Hump Distribution')

    for i in range(3):
        ax[i].legend()
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('Probability Density (normalized)')


  
    return f
