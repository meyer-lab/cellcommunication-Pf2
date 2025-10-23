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


def calculate_KL_divergence_1D(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Calculate KL divergence between two 1D distributions using kernel density estimation.
    
    :param dist1: First distribution (1D array)
    :param dist2: Second distribution (1D array)
    :return: KL divergence value
    """
    # Use kernel density estimation to estimate probability distributions
    bw_method = "scott"
    kde1 = KernelDensity(bandwidth=bw_method).fit(dist1.reshape(-1, 1))
    kde2 = KernelDensity(bandwidth=bw_method).fit(dist2.reshape(-1, 1))
    
    # Create evaluation points across the range of both distributions
    min_val = min(dist1.min(), dist2.min()) - 0.5
    max_val = max(dist1.max(), dist2.max()) + 0.5
    eval_points = np.linspace(min_val, max_val, 1000).reshape(-1, 1)
    
    # Get probability densities
    probs1 = np.exp(kde1.score_samples(eval_points))
    probs2 = np.exp(kde2.score_samples(eval_points))
    
    # Calculate symmetric KL divergence
    kl1 = stats.entropy(probs2 + 1e-10, probs1 + 1e-10, base=2)
    kl2 = stats.entropy(probs1 + 1e-10, probs2 + 1e-10, base=2)
    
    return (kl1 + kl2) / 2


def calculate_wasserstein_distance(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance) between two 1D distributions.
    
    :param dist1: First distribution (1D array)
    :param dist2: Second distribution (1D array)
    :return: Wasserstein distance
    """
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(dist1, dist2)


def component_distribution_analysis(original_components: np.ndarray, 
                                  resampled_components: np.ndarray) -> dict:
    """
    Compare component distributions using multiple distance metrics.
    
    :param original_components: Original component matrix (cells x components)
    :param resampled_components: Resampled component matrix (cells x components)
    :return: Dictionary with comparison results
    """
    n_orig_comp = original_components.shape[1]
    n_resamp_comp = resampled_components.shape[1]
    
    # Initialize result matrices
    ks_pvalues = np.zeros((n_orig_comp, n_resamp_comp))
    ks_statistics = np.zeros((n_orig_comp, n_resamp_comp))
    kl_divergences = np.zeros((n_orig_comp, n_resamp_comp))
    wasserstein_distances = np.zeros((n_orig_comp, n_resamp_comp))
    
    # Calculate all pairwise comparisons
    for i in range(n_orig_comp):
        for j in range(n_resamp_comp):
            orig_dist = original_components[:, i]
            resamp_dist = resampled_components[:, j]
            
            # KS test
            ks_stat, p_val = ks_2samp(orig_dist, resamp_dist)
            ks_pvalues[i, j] = p_val
            ks_statistics[i, j] = ks_stat
            
            # KL divergence
            kl_div = calculate_KL_divergence_1D(orig_dist, resamp_dist)
            kl_divergences[i, j] = kl_div
            
            # Wasserstein distance
            wasserstein_dist = calculate_wasserstein_distance(orig_dist, resamp_dist)
            wasserstein_distances[i, j] = wasserstein_dist
    
    # Find best matches for each original component using different metrics
    best_matches_ks = np.argmax(ks_pvalues, axis=1)  # Highest p-value = most similar
    best_matches_kl = np.argmin(kl_divergences, axis=1)  # Lowest KL = most similar
    best_matches_wasserstein = np.argmin(wasserstein_distances, axis=1)  # Lowest distance = most similar
    
    return {
        'ks_pvalues': ks_pvalues,
        'ks_statistics': ks_statistics,
        'kl_divergences': kl_divergences,
        'wasserstein_distances': wasserstein_distances,
        'best_matches_ks': best_matches_ks,
        'best_matches_kl': best_matches_kl,
        'best_matches_wasserstein': best_matches_wasserstein,
        'similarity_scores': 1 - ks_statistics[np.arange(n_orig_comp), best_matches_ks]
    } 

def makeFigure():
    ax, f = getSetup((10, 6), (2, 2))  # Simpler 2x2 layout
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
        sg_index = resample_X.obs["condition_unique_idxs"]
        
        print(np.shape(XX))
        print(np.shape(resample_X))
        # print(np.shape(resampled_projected_matrices))
        resampled_projections = np.concatenate(resampled_projected_matrices, axis=0)

        print(np.shape(resampled_projections))
        resample_X.obsm["sc_B"] = resampled_projections @ resampled_cp_factors[1]
        resample_X.obsm["rc_C"] = resampled_projections @ resampled_cp_factors[2]
       

    # Perform KS test for ALL original components vs ALL resampled components
    original_sc_B = XX.obsm["sc_B"]
    resampled_sc_B = resample_X.obsm["sc_B"]
    
    print(f"Original sc_B shape: {original_sc_B.shape}")
    print(f"Resampled sc_B shape: {resampled_sc_B.shape}")
    
    # Subsample to 100 cells for comparison to reduce computational load
    n_subsample = 100
    
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
    
    n_original_components = original_sc_B_sub.shape[1]
    n_resampled_components = resampled_sc_B_sub.shape[1]
    
    # Use comprehensive distribution analysis
    analysis_results = component_distribution_analysis(original_sc_B_sub, resampled_sc_B_sub)
    
    # Extract results
    ks_pvalue_matrix = analysis_results['ks_pvalues']
    ks_statistic_matrix = analysis_results['ks_statistics']
    kl_divergences = analysis_results['kl_divergences']
    wasserstein_distances = analysis_results['wasserstein_distances']
    best_matches = analysis_results['best_matches_ks']
    similarity_scores = analysis_results['similarity_scores']
    best_pvalues = ks_pvalue_matrix[np.arange(n_original_components), best_matches]
    
    # Panel A: Similarity scores (KS-based)
    bars = ax[0].bar(range(n_original_components), similarity_scores, 
                        color='steelblue', alpha=0.8)
    ax[0].set_xlabel('Original Component')
    ax[0].set_ylabel('Similarity Score (0-1)')
    ax[0].set_title('Component Similarity (KS-based)\n(1.0 = Identical Distribution)', 
                       fontsize=12, fontweight='bold')
    ax[0].set_xticks(range(n_original_components))
    ax[0].set_xticklabels([f'C{i+1}' for i in range(n_original_components)])
    ax[0].set_ylim(0, 1)
    ax[0].grid(True, alpha=0.3)
    
    # Add clear value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, similarity_scores)):
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{sim:.3f}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=12, color='darkblue')
    
    # Panel B: KL divergence comparison
    best_kl_values = kl_divergences[np.arange(n_original_components), analysis_results['best_matches_kl']]
    bars_kl = ax[1].bar(range(n_original_components), best_kl_values, 
                        color='orange', alpha=0.8)
    ax[1].set_xlabel('Original Component')
    ax[1].set_ylabel('KL Divergence')
    ax[1].set_title('KL Divergence (Best Matches)\n(Lower = More Similar)', 
                    fontsize=12, fontweight='bold')
    ax[1].set_xticks(range(n_original_components))
    ax[1].set_xticklabels([f'C{i+1}' for i in range(n_original_components)])
    ax[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, kl_val) in enumerate(zip(bars_kl, best_kl_values)):
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                     f'{kl_val:.2f}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, color='darkorange')
    
    # Panel C: Wasserstein distance comparison
    best_wasserstein_values = wasserstein_distances[np.arange(n_original_components), analysis_results['best_matches_wasserstein']]
    bars_w = ax[2].bar(range(n_original_components), best_wasserstein_values, 
                       color='green', alpha=0.8)
    ax[2].set_xlabel('Original Component')
    ax[2].set_ylabel('Wasserstein Distance')
    ax[2].set_title('Wasserstein Distance (Best Matches)\n(Lower = More Similar)', 
                    fontsize=12, fontweight='bold')
    ax[2].set_xticks(range(n_original_components))
    ax[2].set_xticklabels([f'C{i+1}' for i in range(n_original_components)])
    ax[2].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, w_val) in enumerate(zip(bars_w, best_wasserstein_values)):
        height = bar.get_height()
        ax[2].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                     f'{w_val:.2f}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, color='darkgreen')
    
    # Panel D: Comprehensive comparison table
    ax[3].axis('off')
    
    # Create comparison text
    comparison_text = "MULTI-METRIC SIMILARITY ANALYSIS:\n\n"
    comparison_text += f"{'Comp':<6} {'KS Match':<9} {'KL Match':<9} {'WS Match':<9} {'KS Sim':<8} {'KL Div':<8} {'WS Dist'}\n"
    comparison_text += "-" * 70 + "\n"
    
    for i in range(n_original_components):
        ks_match = analysis_results['best_matches_ks'][i] + 1
        kl_match = analysis_results['best_matches_kl'][i] + 1
        ws_match = analysis_results['best_matches_wasserstein'][i] + 1
        
        comparison_text += f"C{i+1:<5} R{ks_match:<8} R{kl_match:<8} R{ws_match:<8} "
        comparison_text += f"{similarity_scores[i]:<8.3f} {best_kl_values[i]:<8.2f} {best_wasserstein_values[i]:<8.2f}\n"
    
    # Add summary statistics
    comparison_text += f"\nSUMMARY:\n"
    comparison_text += f"• Avg KS Similarity: {np.mean(similarity_scores):.3f}\n"
    comparison_text += f"• Avg KL Divergence: {np.mean(best_kl_values):.3f}\n"
    comparison_text += f"• Avg Wasserstein: {np.mean(best_wasserstein_values):.3f}\n"
    
    # Check agreement between metrics
    agreement_ks_kl = sum(analysis_results['best_matches_ks'] == analysis_results['best_matches_kl'])
    agreement_ks_ws = sum(analysis_results['best_matches_ks'] == analysis_results['best_matches_wasserstein'])
    comparison_text += f"• KS-KL Agreement: {agreement_ks_kl}/{n_original_components}\n"
    comparison_text += f"• KS-Wasserstein Agreement: {agreement_ks_ws}/{n_original_components}\n"
    
    ax[3].text(0.05, 0.95, comparison_text, transform=ax[3].transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax[3].set_title('Multi-Metric Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Print comprehensive results to console
    print("\n" + "="*80)
    print("MULTI-METRIC COMPONENT DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"{'Comp':<6} {'KS Best':<9} {'KL Best':<9} {'WS Best':<9} {'KS Sim':<9} {'KL Div':<9} {'WS Dist':<9} {'P-Value'}")
    print("-"*80)
    
    for i in range(n_original_components):
        ks_match = analysis_results['best_matches_ks'][i] + 1
        kl_match = analysis_results['best_matches_kl'][i] + 1
        ws_match = analysis_results['best_matches_wasserstein'][i] + 1
        
        print(f"C{i+1:<5} R{ks_match:<8} R{kl_match:<8} R{ws_match:<8} "
              f"{similarity_scores[i]:<9.3f} {best_kl_values[i]:<9.3f} "
              f"{best_wasserstein_values[i]:<9.3f} {best_pvalues[i]:<9.2e}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"• Average KS Similarity: {np.mean(similarity_scores):.3f}")
    print(f"• Average KL Divergence: {np.mean(best_kl_values):.3f}")
    print(f"• Average Wasserstein Distance: {np.mean(best_wasserstein_values):.3f}")
    
    # Metric agreement analysis
    agreement_ks_kl = sum(analysis_results['best_matches_ks'] == analysis_results['best_matches_kl'])
    agreement_ks_ws = sum(analysis_results['best_matches_ks'] == analysis_results['best_matches_wasserstein'])
    agreement_kl_ws = sum(analysis_results['best_matches_kl'] == analysis_results['best_matches_wasserstein'])
    
    print(f"\nMETRIC AGREEMENT:")
    print(f"• KS-KL Agreement: {agreement_ks_kl}/{n_original_components} ({100*agreement_ks_kl/n_original_components:.1f}%)")
    print(f"• KS-Wasserstein Agreement: {agreement_ks_ws}/{n_original_components} ({100*agreement_ks_ws/n_original_components:.1f}%)")
    print(f"• KL-Wasserstein Agreement: {agreement_kl_ws}/{n_original_components} ({100*agreement_kl_ws/n_original_components:.1f}%)")
    
    # Best overall matches
    best_overall_ks = np.argmax(similarity_scores)
    best_overall_kl = np.argmin(best_kl_values)
    best_overall_ws = np.argmin(best_wasserstein_values)
    
    print(f"\nBEST OVERALL MATCHES:")
    print(f"• KS: Component {best_overall_ks+1} → Resample {analysis_results['best_matches_ks'][best_overall_ks]+1} (sim: {similarity_scores[best_overall_ks]:.3f})")
    print(f"• KL: Component {best_overall_kl+1} → Resample {analysis_results['best_matches_kl'][best_overall_kl]+1} (div: {best_kl_values[best_overall_kl]:.3f})")
    print(f"• Wasserstein: Component {best_overall_ws+1} → Resample {analysis_results['best_matches_wasserstein'][best_overall_ws]+1} (dist: {best_wasserstein_values[best_overall_ws]:.3f})")
    print("="*80)
    
    return f

