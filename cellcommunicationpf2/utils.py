import anndata
import numpy as np
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms

def resample(data: anndata.AnnData, random_seed: int = None) -> anndata.AnnData:
    """Perform stratified bootstrap sampling by resampling cells within each sample.

    This maintains the same number of cells per sample in the resampled dataset.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    resampled_indices = []
    for sample in data.obs["sample"].unique():
        # Get indices of cells in this sample
        sample_cell_indices = np.where(data.obs["sample"] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Sample with replacement within this sample's cells
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData from resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


def calculate_fms(A: anndata.AnnData, B: anndata.AnnData) -> float:
    """Calculate FMS between two CC-PF2 decompositions stored in AnnData objects.

    Skips comparison of sender/receiver factors (modes 1 and 2) as they are most variable.
    """
    factors_A = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.uns["Pf2_C"], A.uns["Pf2_D"]]
    A_CP = CPTensor((A.uns["Pf2_weights"], factors_A))

    factors_B = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.uns["Pf2_C"], B.uns["Pf2_D"]]
    B_CP = CPTensor((B.uns["Pf2_weights"], factors_B))

    return fms(A_CP, B_CP, consider_weights=False, skip_mode=[1, 2])
