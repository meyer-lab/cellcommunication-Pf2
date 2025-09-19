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

    # # Import and prepare data
    # adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    # lr_pairs = import_ligand_receptor_pairs()

    # # Add numerical indices for each patient sample, which is the primary condition
    # condition_column = "sample"
    # adata_filtered = add_cond_idxs(adata, condition_column)

    # # Create a mapping from each sample to its corresponding condition (e.g., 'severe')
    # # This will be used for grouping and coloring the heatmap
    # group_col = "condition"
    # sample_to_group = adata_filtered.obs.drop_duplicates(
    #     subset=[condition_column, group_col]
    # ).set_index(condition_column)[group_col]

    # # Parameters for CCC-RISE
    # rise_rank = 35
    # cp_rank = 8
    # n_iter_max = 10000
    # tol = 1e-9

    # print(f"Running CCC-RISE with rank={rise_rank} and cp_rank={cp_rank}...")
    # adata_filtered, _ = run_ccc_rise_workflow(
    #     adata_filtered,
    #     rise_rank=rise_rank,
    #     lr_pairs=lr_pairs,
    #     condition_column=condition_column,
    #     cp_rank=cp_rank,
    #     n_iter_max=n_iter_max,
    #     tol=tol,
    #     complex_sep="&",
    
    # )
    # # Save anndata object with results
    # adata_filtered.write_h5ad("cellcommunicationpf2/data/bal/bal.h5ad")
    
    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]

    # Factor 0: Patient Conditions (Samples)
    plot_condition_factors(
        X,
        ax[0],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,  # Sort samples by their condition group
        normalize=True,
    )
    ax[0].set_title("Factor 0: Patient Condition")

    # Factor 1: Sender Cell Eigenstates
    plot_eigenstate_factors(X, ax[1], factor_type="B")
    ax[1].set_title("Factor 1: Sender Cell Eigen-state")
    rotate_yaxis(ax[1], rotation=0)

    # Factor 2: Receiver Cell Eigenstates
    plot_eigenstate_factors(X, ax[2], factor_type="C")
    ax[2].set_title("Factor 2: Receiver Cell Eigen-state")
    rotate_yaxis(ax[2], rotation=0)

    # Factor 3: Ligand-Receptor Pairs
    plot_lr_factors(X, ax[3], trim=True, weight=0.06)
    ax[3].set_title("Factor 3: LR Pair")

    return f