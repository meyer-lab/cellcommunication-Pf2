"""
Figure A3a: CCC-RISE on BALF COVID-19 data.
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
from ..utils import run_ccc_rise_workflow
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
import anndata
from .commonFuncs.plotGeneral import rotate_yaxis

def makeFigure():
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    
    A_factor = X.uns["A"]
    
    