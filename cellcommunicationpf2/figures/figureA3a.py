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
from ..import_data import prepare_dataset

def makeFigure():
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)
    
    
    
    X = import_alad(0.1, normalize=True)
    print(X)
    # X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    # X = import_lupus(geneThreshold=0.1)
    print(X)
    print(X.var_names)
    lr_pairs = import_ligand_receptor_pairs()
    
    # Find overlap between lr_pairs and X.var_names
    lr_pairs = lr_pairs[
        (lr_pairs["ligand"].isin(X.var_names)) & (lr_pairs["receptor"].isin(X.var_names))
    ].reset_index(drop=True)
    print(lr_pairs)
    print(f"Number of LR pairs after filtering: {len(lr_pairs)}")
    
    


    
    # condition_column = "Condition"
    # # Parameters for CCC-RISE
    # rise_rank = 30
    # cp_rank = 10
    # n_iter_max = 10000
    # tol = 1e-9

    # print(f"Running CCC-RISE with rank={rise_rank} and cp_rank={cp_rank}...")
    # X, _ = run_ccc_rise_workflow(
    #     X,
    #     rise_rank=rise_rank,
    #     lr_pairs=lr_pairs,
    #     condition_column=condition_column,
    #     cp_rank=cp_rank,
    #     n_iter_max=n_iter_max,
    #     tol=tol,
    #     complex_sep="&",
    # )
    # # # Save anndata object with results
    # # adata_filtered.write_h5ad("cellcommunicationpf2/data/bal/bal.h5ad")
    # # Factor 0: Patient Conditions (Samples)
    # plot_condition_factors(
    #     X,
    #     ax[0],
    #     cond=condition_column,
    #     # cond_group_labels=sample_to_group,
    #     group_cond=True,  # Sort samples by their condition group
    #     normalize=True,
    # )
    # ax[0].set_title("Factor 0: Patient Condition")

    # # Factor 1: Sender Cell Eigenstates
    # plot_eigenstate_factors(X, ax[1], factor_type="B")
    # ax[1].set_title("Factor 1: Sender Cell Eigen-state")
    # rotate_yaxis(ax[1], rotation=0)

    # # Factor 2: Receiver Cell Eigenstates
    # plot_eigenstate_factors(X, ax[2], factor_type="C")
    # ax[2].set_title("Factor 2: Receiver Cell Eigen-state")
    # rotate_yaxis(ax[2], rotation=0)

    # # Factor 3: Ligand-Receptor Pairs
    # plot_lr_factors(X, ax[3], trim=True, weight=0.06)
    # ax[3].set_title("Factor 3: LR Pair")

    return f



def import_alad(
    gene_threshold: float = 0.1, normalize: bool = True
):
    """Generate data for ULTRA analysis, filtering and scaling as needed.
    patient ids: dsco_id
    predictor: ALADstatus
    """
 
    data = anndata.read_h5ad("/opt/BAL-scRNAseq-raw.h5ad", backed="r")
    print(data)
    print(data.X)
    print(data.obs["dsco_id"].unique())
    print(data.obs["danny_broad_annotations"].unique())
    print(data.obs["danny_narrow_annotations"].unique())
    print(data.obs["cell_type_group"].unique())

    # Filter out controls
    # data = data[data.obs["ALADstatus"] != "control", :]

    # normalize by read depth, transform
    data = prepare_dataset(data, "dsco_id", geneThreshold=gene_threshold, normalize=normalize)

    return data


def import_lupus(geneThreshold: float = 0.1) -> anndata.AnnData:
    """Import Lupus PBMC dataset.

    -- columns from observation data:
    {'batch_cov': POOL (1-23) cell was processed in,
    'ind_cov': patient cell was derived from,
    'Processing_Cohort': BATCH (1-4) cell was derived from,
    'louvain': louvain cluster group assignment,
    'cg_cov': broad cell type,
    'ct_cov': lymphocyte-specific cell type,
    'L3': marks a balanced subset of batch 4 used for model training,
    'ind_cov_batch_cov': combination of patient and pool, proxy for sample ID,
    'Age':	age of patient,
    'Sex': sex of patient,
    'pop_cov': ancestry of patient,
    'Status': SLE status: healthy, managed, treated, or flare,
    'SLE_status': SLE status: healthy or SLE}

    """
    X = anndata.read_h5ad("/opt/andrew/lupus/raw_lupus.h5ad")

    protein = anndata.read_h5ad("/opt/andrew/lupus/Lupus_study_protein_adjusted.h5ad")
    protein_df = protein.to_df()

    # Rename columns
    X.obs = X.obs.rename(
        {
            "batch_cov": "pool",
            "ind_cov": "patient",
            "cg_cov": "Cell Type",
            "ct_cov": "cell_type_lympho",
            "ind_cov_batch_cov": "Condition",
            "Age": "age",
            "Sex": "sex",
            "pop_cov": "ancestry",
        },
        axis=1,
    )

    X.obs = X.obs.merge(protein_df, how="left", left_index=True, right_index=True)
    print("ehhlo")
    # Get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (Only 3 cells)
    X = X[X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]
    # condition_column = "Condition"
    # X = add_cond_idxs(X, condition_column)
    print("ehhlo")
    return prepare_dataset(X, "Condition", geneThreshold=geneThreshold, normalize=True)