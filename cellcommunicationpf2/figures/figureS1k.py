"""
Figure S1c_d: FMS across CPD ranks (only) for COVID-19
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy import stats
from sklearn.neighbors import KernelDensity
from ..import_data import add_cond_idxs, import_balf_covid, import_ligand_receptor_pairs
from .common import getSetup, subplotLabel
from ..tensor import run_fms_r2x_analysis, calculate_interaction_tensor
from ..utils import resample
from tensorly.decomposition import parafac
import anndata
from scipy.stats import pearsonr
import ot


def distance_metric(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Calculate KL divergence between two 1D distributions using kernel density estimation.
    
    :param dist1: First distribution (1D array)
    :param dist2: Second distribution (1D array)
    :return: KL divergence value
    """
    # Use kernel density estimation to estimate probability distributions
    assert dist1.shape[1] == dist2.shape[1], "Both distributions must have the same number of dimensions"

    n_dim = dist1.shape[1]

    # Estimate the n-dimensional probability distributions
    bw_method = "scott"
    kde1 = KernelDensity(atol=1e-9, rtol=1e-9, bandwidth=bw_method).fit(dist1)
    kde2 = KernelDensity(atol=1e-9, rtol=1e-9, bandwidth=bw_method).fit(dist2)

    # Compare over the entire distribution space by looking at the global max/min
    min_abun = np.minimum(dist1.min(axis=0), dist2.min(axis=0)) - 0.5
    max_abun = np.maximum(dist1.max(axis=0), dist2.max(axis=0)) + 0.5

    # Create a mesh grid for n-dimensional comparison
    grids = np.meshgrid(
        *[np.linspace(min_abun[i], max_abun[i], 10) for i in range(n_dim)]
    )
    grids = np.stack([grid.flatten() for grid in grids], axis=-1)

    # Calculate the probabilities (log-likelihood) of all combinations in the mesh grid
    dist1_probs = np.exp(kde1.score_samples(grids))
    dist2_probs = np.exp(kde2.score_samples(grids))

    # Calculate KL Divergence
    KL_div_val_1 = stats.entropy(dist2_probs + 1e-200, dist1_probs + 1e-200, base=2)
    KL_div_val_2 = stats.entropy(dist1_probs + 1e-200, dist2_probs + 1e-200, base=2)
    KL_div_val = float((KL_div_val_1 + KL_div_val_2) / 2)

    # Calculate EMD
    M = ot.dist(dist1, dist2, metric="euclidean")
    EMD_val = ot.emd2([], [], M, numItermax=int(1e9))
    
    return KL_div_val, EMD_val



def makeFigure():
    ax, f = getSetup((8, 8), (2, 2))  # Larger layout for heatmaps
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)

    rise_rank = 35
    cpd_rank = 8
    repeats = 1
    
    XX = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    for i in range(repeats):
        resample_X = resample(X_filtered, condition_name=condition_column)
        resampled_interaction_tensor, resampled_projected_matrices = calculate_interaction_tensor(
            resample_X,
            lr_pairs,
            rise_rank=rise_rank  # Use rise_rank instead of cpd_rank
        )
        _, resampled_cp_factors = parafac(
                tensor=resampled_interaction_tensor,
                rank=cpd_rank,
                n_iter_max=1000,
                init="random",
                normalize_factors=True,
            )
        resampled_projections = np.concatenate(resampled_projected_matrices, axis=0)
        resample_X.obsm["sc_B"] = resampled_projections @ resampled_cp_factors[1]
        resample_X.obsm["rc_C"] = resampled_projections @ resampled_cp_factors[2]
       

    # Perform KS test for ALL original components vs ALL resampled components
    original_sc_B = XX.obsm["sc_B"]
    resampled_sc_B = resample_X.obsm["sc_B"]
    
    print(f"Original sc_B shape: {original_sc_B.shape}")
    print(f"Resampled sc_B shape: {resampled_sc_B.shape}")
    
    # Subsample to 100 cells for comparison to reduce computational load

    n_subsample = 5000
    
    # Randomly subsample from original data
    if original_sc_B.shape[0] > n_subsample:
        orig_indices = np.random.choice(original_sc_B.shape[0], n_subsample, replace=False)
        original_sc_B_sub = original_sc_B[orig_indices, :]
        print(f"Subsampled original to {n_subsample} cells")
    else:
        original_sc_B_sub = original_sc_B
        print(f"Using all {original_sc_B.shape[0]} original cells")
    
    # Randomly subsample from resampled data
    if resampled_sc_B.shape[0] > n_subsample:
        resamp_indices = np.random.choice(resampled_sc_B.shape[0], n_subsample, replace=False)
        resampled_sc_B_sub = resampled_sc_B[resamp_indices, :]
        print(f"Subsampled resampled to {n_subsample} cells")
    else:
        resampled_sc_B_sub = resampled_sc_B
        print(f"Using all {resampled_sc_B.shape[0]} resampled cells")
    
    print(f"Final comparison shapes: Original {original_sc_B_sub.shape}, Resampled {resampled_sc_B_sub.shape}")

    metric_kl = np.zeros((original_sc_B_sub.shape[1], resampled_sc_B_sub.shape[1]))
    metric_emd = np.zeros((original_sc_B_sub.shape[1], resampled_sc_B_sub.shape[1]))
    metric_ttest = np.zeros((original_sc_B_sub.shape[1], resampled_sc_B_sub.shape[1]))
    metric_kstest = np.zeros((original_sc_B_sub.shape[1], resampled_sc_B_sub.shape[1]))
    for i in range(original_sc_B_sub.shape[1]):
        for j in range(resampled_sc_B_sub.shape[1]):
            
                  
            orig_dist = original_sc_B_sub[:, i].reshape(-1, 1)  # Reshape to 2D for KDE
            resamp_dist = resampled_sc_B_sub[:, j].reshape(-1, 1)  # Reshape to 2D for KDE
            
            metric_ttest[i, j] = stats.ttest_ind(orig_dist, resamp_dist).pvalue
            metric_kstest[i, j] = ks_2samp(orig_dist.flatten(), resamp_dist.flatten()).pvalue
      
            kl, emd = distance_metric(orig_dist, resamp_dist)
            metric_kl[i, j] = kl
            metric_emd[i, j] = emd


    sns.heatmap(metric_kl, ax=ax[0], cmap="viridis")
    ax[0].set_title("KL Divergence between Original and Resampled sc_B Components")
    ax[0].set_xlabel("Resampled sc_B Components")
    ax[0].set_ylabel("Original sc_B Components")
    
    sns.heatmap(metric_emd, ax=ax[1], cmap="viridis")
    ax[1].set_title("EMD between Original and Resampled sc_B Components")
    ax[1].set_xlabel("Resampled sc_B Components")
    ax[1].set_ylabel("Original sc_B Components")    
    
    sns.heatmap(metric_ttest, ax=ax[2], cmap="viridis")
    ax[2].set_title("T-test p-values between Original and Resampled sc_B Components")
    ax[2].set_xlabel("Resampled sc_B Components")
    ax[2].set_ylabel("Original sc_B Components")    
    
    sns.heatmap(metric_kstest, ax=ax[3], cmap="viridis")
    ax[3].set_title("KS-test p-values between Original and Resampled sc_B Components")
    ax[3].set_xlabel("Resampled sc_B Components")
    ax[3].set_ylabel("Original sc_B Components")
    
    # Log coloring for p-value heatmaps
    for a in [ax[2], ax[3]]:
        norm = plt.Normalize(vmin=1e-10, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        a.figure.colorbar(sm, ax=a, orientation='vertical', label='p-value (log scale)')
        

    
    

    
    return f

