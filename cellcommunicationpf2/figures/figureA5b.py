"""
Figure A5b: CCC-RISE on BALF COVID-19 data. Showing ligand-receptor factors.
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
from ..utils import run_cc_pf2_workflow
from .commonFuncs.plotFactors import (
    plot_lr_factors_partial
)

def makeFigure():
    ax, f = getSetup((12, 12), (5, 4))
    subplotLabel(ax)

    # Import and prepare data
    adata = import_balf_covid(gene_threshold=0.01, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X = add_cond_idxs(adata, condition_column)

    # Parameters for CCC-RISE
    rise_rank = 30
    cp_rank = 10
    n_iter_max = 1000
    tol = 1e-9
    random_state = 42
    
    print(f"Running CCC-RISE with rank={rise_rank} and cp_rank={cp_rank}...")
    X, r2x = run_cc_pf2_workflow(
        X,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
        complex_sep="&"
        
    )

    print(f"CCC-RISE decomposition R2X: {r2x:.4f}")
    
    for i in range(cp_rank):
        plot_lr_factors_partial(X, i, ax[2*i], geneAmount=10, top=True)
        plot_lr_factors_partial(X, i, ax[2*i+1], geneAmount=10, top=False)
        
    return f