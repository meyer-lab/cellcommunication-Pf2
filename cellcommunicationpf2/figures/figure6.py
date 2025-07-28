"""
Figure 6: CP Rank Selection Analysis

This figure demonstrates how R²X changes with increasing CP rank
while keeping the RISE rank fixed at 30.
"""

from matplotlib import pyplot as plt

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import subplotLabel
from ..utils import run_cc_pf2_workflow

def makeFigure():
    """Generate Figure 6 showing CP rank selection analysis with fixed RISE rank."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    subplotLabel([ax])

    # Import and prepare data
    print("Importing and preparing data for Figure 6...")
    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")
    lr_pairs = import_ligand_receptor_pairs()
    print(f"Data shape: {adata_filtered.shape}")

    rise_rank = 30
    cp_ranks = list(range(10, 101, 10))
    print(f"Testing CP ranks {cp_ranks[0]}-{cp_ranks[-1]} in steps of 5 (RISE rank fixed at {rise_rank})...")

    n_iter_max = 100
    tol = 1e-6
    random_state = 42

    r2x_results = {}

    for i, cp_rank in enumerate(cp_ranks):
        print(f"Testing CP rank {cp_rank} ({i+1}/{len(cp_ranks)})...")
        try:
            _, r2x = run_cc_pf2_workflow(
                adata_filtered,
                rise_rank=rise_rank,
                lr_pairs=lr_pairs,
                cp_rank=cp_rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
            )
            r2x_results[cp_rank] = r2x
            print(f"  CP Rank {cp_rank}: R²X = {r2x:.6f}")

        except Exception as e:
            print(f"  CP Rank {cp_rank}: Failed - {e}")
            r2x_results[cp_rank] = float('nan')

    # Filter valid results and plot
    valid_results = {k: v for k, v in r2x_results.items() if not (v is None or v != v)}
    cp_ranks = sorted(valid_results.keys())
    r2x_values = [valid_results[r] for r in cp_ranks]

    # Plot R²X curve
    ax.plot(cp_ranks, r2x_values, "o-", linewidth=2, markersize=6, color="darkorange")
    ax.set_xlabel("CP Rank", fontsize=14)
    ax.set_ylabel("R²X", fontsize=14)
    ax.set_title(f"CP Rank Selection (RISE Rank = {rise_rank})", fontsize=16)
    ax.grid(True, alpha=0.3)

    print("Figure 6 generation complete.")
    return fig