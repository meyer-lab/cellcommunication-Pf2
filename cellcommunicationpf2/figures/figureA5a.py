"""
Figure A3a: CCC-RISE on BALF COVID-19 data.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from ..import_data import (
    add_cond_idxs,
    import_alad,
    import_ligand_receptor_pairs,
)
from ..utils import run_ccc_rise_workflow, correct_conditions
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

    # Import and prepare data
    X = import_alad(gene_threshold=0.001, normalize=True)
    # Remove control samples if any
    X = X[X.obs["ALADstatus"] != "control"]
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)
    
    # Parameters for CCC-RISE
    rise_rank = 20
    cp_rank = 20
    n_iter_max = 10000
    tol = 1e-11

    print(f"Running CCC-RISE with rank={rise_rank} and cp_rank={cp_rank}...")
    X, _ = run_ccc_rise_workflow(
        X,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        condition_column=condition_column,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        complex_sep="&",
    
    )
    # # Save anndata object with results
    # X.write_h5ad("cellcommunicationpf2/alad.h5ad")
    
    
    # X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    condition_column = "dsco_id"
    # X.uns["A"] = correct_conditions(X)


    group_col = "ALADstatus"
    sample_to_group = X.obs.drop_duplicates(
        subset=[condition_column, group_col]
    ).set_index(condition_column)[group_col]
    
    print(sample_to_group)

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
    
    
    severity_map = {k: i+1 for i, k in enumerate(np.unique(sample_to_group))}
    severity_numeric = sample_to_group.map(severity_map).values
    
    cmp = X.uns["A"].shape[1]
    pearson_corr = np.zeros((cmp))
    for i in range(cmp):
        pearson_corr[i] = pearsonr(X.uns["A"][:, i-1], severity_numeric)[0]
    pearson_df = pd.DataFrame(pearson_corr, index=[f"{i+1}" for i in range(cmp)], columns=["Pearson Correlation"])
    pearson_df["Component"] = pearson_df.index
    print(pearson_df)
    # sns.barplot(x="Component", y="Pearson Correlation", data=pearson_df, ax=ax[0], color="black")
    # ax[0].set_ylim(-0.1, 1)

    return f