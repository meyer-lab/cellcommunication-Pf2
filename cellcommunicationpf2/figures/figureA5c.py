"""
Figure A5c: CCC-RISE on BALF COVID-19 data. Showing weighted sender and receiver cell factors.
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
from .commonFuncs.plotPaCMAP import (
    plot_wc_per_celltype,
    plot_wc_pacmap
)   

def makeFigure():
    ax, f = getSetup((18, 18), (5, 4))
    subplotLabel(ax)

    # Import and prepare data
    adata = import_balf_covid(gene_threshold=0, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X = add_cond_idxs(adata, condition_column)

    # Parameters for CCC-RISE
    rise_rank = 30
    cp_rank = 10
    n_iter_max = 100
    tol = 1e-6
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
    
    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i], cellType="celltype", factor_matrix="B")
        
    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i+cp_rank], cellType="celltype", factor_matrix="C")
        
    for i in range(cp_rank):
        plot_wc_pacmap(X, i + 1, ax[i], factor_matrix="B")
        
    for i in range(cp_rank):
        plot_wc_pacmap(X, i + 1, ax[i+cp_rank], factor_matrix="C")

    return f