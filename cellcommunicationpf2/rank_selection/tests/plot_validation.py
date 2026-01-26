"""Validate rank selection using factorize-reconstruct approach."""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/nthomas/cellcommunication-Pf2")

from cellcommunicationpf2.import_data import import_balf_covid, add_cond_idxs
from cellcommunicationpf2.rank_selection import run_rank_selection
from tensorly.parafac2_tensor import parafac2_to_slices
from kneed import KneeLocator
from parafac2.parafac2 import parafac2_nd
from scipy.sparse import csr_array

TARGET_CELLS_PER_COND = 2500
TARGET_N_CONDITIONS = 8
TARGET_N_GENES = 200

print("Loading real scRNA-seq data...")
adata_full = import_balf_covid()
print(f"Full data: {adata_full.shape}")


def create_known_rank_data(adata_full, true_rank, n_conditions=8, cells_per_cond=100, n_genes=200):
    """Factorize real data at known rank, reconstruct, then subsample."""
    print(f"\n=== Creating known-rank-{true_rank} data ===")
    
    conditions = adata_full.obs['sample'].unique()[:n_conditions]
    print(f"Using {len(conditions)} conditions")
    
    indices = []
    for cond in conditions:
        cond_idx = np.where(adata_full.obs['sample'] == cond)[0]
        n_sample = min(cells_per_cond, len(cond_idx))
        np.random.seed(42)
        sampled = np.random.choice(cond_idx, size=n_sample, replace=False)
        indices.extend(sampled)
    
    adata_sub = adata_full[indices].copy()
    gene_means = np.array(adata_sub.X.mean(axis=0)).flatten()
    top_genes = np.argsort(gene_means)[-n_genes:]
    adata_sub = adata_sub[:, top_genes].copy()
    adata_sub = add_cond_idxs(adata_sub, "sample")
    
    print(f"Subsampled: {adata_sub.shape}")
    
    print(f"Factorizing at rank {true_rank}...")
    pf2_output, r2x = parafac2_nd(
        adata_sub, rank=true_rank, n_iter_max=100, tol=1e-6, random_state=42
    )
    print(f"R2X at rank {true_rank}: {r2x:.4f}")
    
    weights, factors, projections = pf2_output
    slices = parafac2_to_slices((weights, factors, projections))
    
    condition_idxs = adata_sub.obs["condition_unique_idxs"].values
    n_genes_out = factors[2].shape[0]
    
    X_recon = np.zeros((adata_sub.n_obs, n_genes_out), dtype=np.float32)
    for cond_idx in np.unique(condition_idxs):
        mask = (condition_idxs == cond_idx)
        X_recon[mask, :] = slices[cond_idx]
    
    adata_recon = adata_sub.copy()
    adata_recon.X = csr_array(X_recon)
    
    return adata_recon, r2x


def find_elbow(ranks, scores):
    """Find elbow point using Kneedle algorithm."""
    try:
        kneedle = KneeLocator(ranks, scores, curve='convex', direction='decreasing')
        return kneedle.knee if kneedle.knee else ranks[0]
    except:
        if len(scores) < 3:
            return ranks[0]
        second_diffs = np.diff(np.diff(scores))
        return ranks[np.argmax(second_diffs) + 1]


true_ranks = [5, 10, 15]
results_all = {}

for tr in true_ranks:
    adata_known_rank, r2x_fit = create_known_rank_data(
        adata_full, tr, 
        n_conditions=TARGET_N_CONDITIONS,
        cells_per_cond=TARGET_CELLS_PER_COND,
        n_genes=TARGET_N_GENES
    )
    
    ranks_to_test = sorted(set(
        list(range(max(2, tr - 5), tr + 1)) + [tr + 3, tr + 6, tr + 10]
    ))
    ranks_to_test = [r for r in ranks_to_test if 2 <= r <= 100]
    
    print(f"\nRunning rank selection on known-rank-{tr} data...")
    print(f"Data shape: {adata_known_rank.shape}, Testing ranks: {ranks_to_test}")
    
    results = run_rank_selection(
        adata_known_rank, ranks=ranks_to_test, condition_key="sample",
        n_folds=5, n_iter_max=100, tol=1e-6, ot_epsilon=0.05, random_state=42
    )
    
    ranks = results["rank"].values
    scores = results["ot_score_mean"].values
    elbow = find_elbow(ranks, scores)
    
    results_all[tr] = {"results": results, "elbow": elbow, "r2x_fit": r2x_fit}
    print(f"True rank: {tr}, Selected (elbow): {elbow}")

# Plot
n_plots = len(true_ranks)
fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
if n_plots == 1:
    axes = [axes]

for ax, tr in zip(axes, true_ranks):
    data = results_all[tr]
    results = data['results']
    
    ranks = results["rank"].values
    scores = results["ot_score_mean"].values
    stds = results["ot_score_std"].values
    
    ax.errorbar(ranks, scores, yerr=stds, marker='o', markersize=7, 
                linewidth=2, capsize=4, color='steelblue', label='OT Score')
    ax.axvline(tr, color='green', linestyle='-', linewidth=3, 
               label=f'True ({tr})', alpha=0.8)
    ax.axvline(data['elbow'], color='red', linestyle='--', linewidth=2,
               label=f'Elbow ({data["elbow"]})', alpha=0.8)
    
    ax.set_xlabel('Rank', fontsize=11, fontweight='bold')
    ax.set_ylabel('OT Score', fontsize=11, fontweight='bold')
    correct = "✓" if data['elbow'] == tr else f"off by {abs(data['elbow']-tr)}"
    ax.set_title(f'True Rank = {tr} ({correct})', fontsize=12, fontweight='bold')
    ax.set_xticks(ranks)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Rank Selection: Factorize-Reconstruct Validation', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = "/home/nthomas/cellcommunication-Pf2/cellcommunicationpf2/rank_selection/output/validation_factorize_reconstruct.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: {output_path}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for tr in true_ranks:
    data = results_all[tr]
    correct = "✓" if data['elbow'] == tr else "✗"
    print(f"{correct} True={tr}, Elbow={data['elbow']}, R2X@fit={data['r2x_fit']:.4f}")

plt.show()
