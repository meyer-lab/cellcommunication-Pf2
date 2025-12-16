"""Generate figures for the CCC-RISE tutorial documentation."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "_static" / "tutorial_images"
output_dir.mkdir(parents=True, exist_ok=True)

# Import CCC-RISE modules
from cellcommunicationpf2.import_data import (
    import_balf_covid,
    add_cond_idxs,
    import_ligand_receptor_pairs,
)
from cellcommunicationpf2.tensor import (
    run_ccc_rise_workflow,
    run_fms_r2x_analysis,
    calculate_interaction_tensor,
)
from cellcommunicationpf2.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
from cellcommunicationpf2.figures.commonFuncs.plotPaCMAP import (
    plot_wc_pacmap,
    plot_labels_pacmap,
)
from cellcommunicationpf2.figures.commonFuncs.plotGeneral import plot_fms_r2x_diff_ranks
from cellcommunicationpf2.utils import (
    expression_product_matrix,
    average_product_matrix_ccc,
)

print("Loading dataset...")
X = import_balf_covid(gene_threshold=0.001, normalize=True)
lr_pairs = import_ligand_receptor_pairs()

# Add numerical indices for each sample (condition)
condition_column = "sample"
X = add_cond_idxs(X, condition_column)

print("\n=== RISE Rank Selection ===")
print("Running RISE FMS/R2X analysis...")
# Use actual RISE rank selection with plot_fms_r2x_diff_ranks
rise_ranks = list(range(5, 11, 5))
runs = 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

print(f"Testing RISE ranks: {rise_ranks[0]}-{rise_ranks[-1]}, {runs} runs...")
plot_fms_r2x_diff_ranks(
    X,
    condition_column,
    axes[0],
    axes[1],
    ranksList=rise_ranks,
    runs=runs
)

axes[0].set_title("RISE: Factor Match Score (FMS)")
axes[0].set_xlabel("RISE Rank")
axes[0].axhline(y=0.6, color='r', linestyle='--', alpha=0.5)

axes[1].set_title("RISE: Variance Explained (R²X)")
axes[1].set_xlabel("RISE Rank")

plt.tight_layout()
plt.savefig(output_dir / "step1_rise_fms_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== CPD Rank Selection ===")
print("Calculating interaction tensor for CPD rank selection...")
rise_rank_for_cpd = 35

# Calculate interaction tensor
print(f"Building interaction tensor with RISE rank={rise_rank_for_cpd}...")
interaction_tensor = calculate_interaction_tensor(X, lr_pairs, rise_rank=rise_rank_for_cpd)
print(f"Interaction tensor shape: {interaction_tensor.shape}")

# Run actual CPD FMS/R2X analysis
cpd_ranks = list(range(1, 4, 2))
runs = 1
print(f"Testing CPD ranks: {cpd_ranks}, {runs} runs...")
cpd_results_df = run_fms_r2x_analysis(
    interaction_tensor,
    rank_list=cpd_ranks,
    runs=runs,
    svd_init="random"
)

# Figure 2: CPD FMS and R2X
print("Generating Figure 2: CPD rank selection...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=cpd_results_df, x='Component', y='FMS', ax=axes[0], marker='o')
axes[0].set_xlabel("CPD Rank")
axes[0].set_ylabel("Factor Match Score (FMS)")
axes[0].set_title(f"CPD: Factor Stability (RISE rank={rise_rank_for_cpd})")
axes[0].axhline(y=0.6, color='r', linestyle='--', label='Stability Threshold')
axes[0].set_ylim(0, 1)
axes[0].legend()

sns.lineplot(data=cpd_results_df, x='Component', y='R2X', ax=axes[1], marker='o', color='orange')
axes[1].set_xlabel("CPD Rank")
axes[1].set_ylabel("R²X (Variance Explained)")
axes[1].set_title(f"CPD: Variance Explained (RISE rank={rise_rank_for_cpd})")

plt.tight_layout()
plt.savefig(output_dir / "step2_cpd_fms_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== Running CCC-RISE Factorization ===")
# Use smaller ranks for faster execution
rise_rank = 20
cp_rank = 5
print(f"Running with rise_rank={rise_rank}, cp_rank={cp_rank}...")
print("Note: Using smaller ranks for demonstration. For production, use ranks from rank selection analysis.")
X, r2x = run_ccc_rise_workflow(
    adata=X,
    rise_rank=rise_rank,
    cp_rank=cp_rank,
    lr_pairs=lr_pairs,
    condition_column="condition_unique_idxs",
    doEmbedding=True,
    random_state=42,
    n_iter_max=100,
    tol=1e-3,
)
print(f"Variance Explained (R²X): {r2x:.4f}")

# Figure 3: All Four Factors
print("Generating Figure 3: All four factors visualization...")
fig, ax = plt.subplots(1, 4, figsize=(20, 8))

# Prepare condition grouping
group_col = "condition"
sample_to_group = X.obs.drop_duplicates(
    subset=[condition_column, group_col]
).set_index(condition_column)[group_col]

# Factor A: Condition factors
plot_condition_factors(
    X,
    ax[0],
    cond=condition_column,
    cond_group_labels=sample_to_group,
    group_cond=True,
    normalize=True
)
ax[0].set_title("Factor A: Condition")

# Factor B: Sender cell eigenstates
plot_eigenstate_factors(X, ax[1], factor_type="B")
ax[1].set_title("Factor B: Sender Cell Eigenstate")

# Factor C: Receiver cell eigenstates
plot_eigenstate_factors(X, ax[2], factor_type="C")
ax[2].set_title("Factor C: Receiver Cell Eigenstate")

# Factor D: Ligand-receptor pairs
plot_lr_factors(X, ax[3], trim=True, weight=0.06)
ax[3].set_title("Factor D: LR Pairs")

plt.tight_layout()
plt.savefig(output_dir / "step3_all_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Detailed Condition Factors
print("Generating Figure 4: Detailed condition factors...")
fig, ax = plt.subplots(figsize=(10, 6))
plot_condition_factors(
    X,
    ax,
    cond=condition_column,
    cond_group_labels=sample_to_group,
    group_cond=True,
    normalize=True
)
ax.set_xlabel("Component", fontsize=12)
ax.set_ylabel("Condition/Sample", fontsize=12)
ax.set_title("Condition Factor Contributions Across Components", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "step4_condition_factors_detailed.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 5: PaCMAP Embeddings
print("Generating Figure 5: PaCMAP embeddings...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color by cell type
if "celltype" in X.obs.columns:
    pal = sns.color_palette(palette="tab20")
    pal = pal.as_hex()
    plot_labels_pacmap(X, labelType="celltype", ax=axes[0], color_key=pal)
    axes[0].set_title("PaCMAP: Colored by Cell Type")

# Color by condition
if group_col in X.obs.columns:
    pal = sns.color_palette("Set1")
    pal = pal.as_hex()
    plot_labels_pacmap(X, labelType=group_col, ax=axes[1], color_key=pal)
    axes[1].set_title("PaCMAP: Colored by Condition")

plt.tight_layout()
plt.savefig(output_dir / "step5_pacmap_embedding.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 6: Component Loadings
print("Generating Figure 6: Component loadings...")
component = 1  # 1-indexed for plot_wc_pacmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sender cell loadings
plot_wc_pacmap(X, component, axes[0], factor_matrix="B", cbarMax=0.3)
axes[0].set_title(f"Component {component}: Sender Cell Loadings")

# Receiver cell loadings
plot_wc_pacmap(X, component, axes[1], factor_matrix="C", cbarMax=0.3)
axes[1].set_title(f"Component {component}: Receiver Cell Loadings")

plt.tight_layout()
plt.savefig(output_dir / "step6_component_loadings.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 8: Categorical Labels
print("Generating Figure 8: Categorical labels...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Cell type
if "celltype" in X.obs.columns:
    pal = sns.color_palette(palette="Set3")
    pal = pal.as_hex()
    plot_labels_pacmap(X, labelType="celltype", ax=axes[0], color_key=pal)
    axes[0].set_title("Cells Colored by Cell Type")

# Condition
if group_col in X.obs.columns:
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]
    pal = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in pal]
    plot_labels_pacmap(X, labelType=group_col, ax=axes[1], color_key=pal)
    axes[1].set_title("Cells Colored by Condition")

# Sample
plot_labels_pacmap(X, labelType=condition_column, ax=axes[2])
axes[2].set_title("Cells Colored by Sample")

plt.tight_layout()
plt.savefig(output_dir / "step8_categorical_labels.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 9: Violin Plot of Weights
print("Generating Figure 9: Violin plot of component weights...")
component = 2  # 0-indexed
if "celltype" in X.obs.columns:
    # Use first available cell type
    cell_type = X.obs["celltype"].cat.categories[0]
    X_celltype = X[X.obs["celltype"] == cell_type]
    sender_weights = X_celltype.obsm["sc_B"][:, component]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(data=sender_weights, ax=ax)
    ax.set_ylim(-0.1, max(sender_weights) + 0.1)
    ax.set_xlabel(f"{cell_type} Weight Distribution")
    ax.set_ylabel(f"Sender Cell Component {component+1} Association")
    ax.set_title(f"Distribution of Component {component+1} Weights in {cell_type}")
    plt.tight_layout()
    plt.savefig(output_dir / "step9_violin_weights.png", dpi=150, bbox_inches="tight")
    plt.close()

# Figure 10: Expression Product Heatmap
print("Generating Figure 10: Expression product heatmap...")
component = 2 # 0-indexed
ligand = "CCL19"
receptor = "CCR7"

if ligand in X.var_names and receptor in X.var_names and "celltype" in X.obs.columns:
    # Get first two available cell types
    celltypes = X.obs["celltype"].cat.categories
    sender_celltype = celltypes[0] if len(celltypes) > 0 else None
    receiver_celltype = celltypes[1] if len(celltypes) > 1 else celltypes[0]
    
    if sender_celltype and receiver_celltype:
        # Filter and sort sender cells
        X_sender = X[X.obs["celltype"] == sender_celltype]
        X_sender = X_sender[np.argsort(-X_sender.obsm["sc_B"][:, component])]
        
        # Filter and sort receiver cells
        X_receiver = X[X.obs["celltype"] == receiver_celltype]
        X_receiver = X_receiver[np.argsort(-X_receiver.obsm["rc_C"][:, component])]
        
        # Calculate expression product
        df = expression_product_matrix(X_sender, X_receiver, ligand, receptor)
        df = average_product_matrix_ccc(df)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(df, ax=ax, cmap="viridis", vmax=0.12)
        ax.set_xlabel(f"Receiver {receiver_celltype}s")
        ax.set_ylabel(f"Sender {sender_celltype}s")
        ax.set_title(f"{ligand}-{receptor} Interaction in Component {component+1}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_dir / "step10_expression_product.png", dpi=150, bbox_inches="tight")
        plt.close()

# Figure 11: Cell Type Comparison
print("Generating Figure 11: Cell type comparison...")
if ligand in X.var_names and receptor in X.var_names and "celltype" in X.obs.columns:
    celltypes = X.obs["celltype"].cat.categories
    if len(celltypes) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Same sender, two different receivers
        X_sender = X[X.obs["celltype"] == celltypes[0]]
        X_sender = X_sender[np.argsort(-X_sender.obsm["sc_B"][:, component])]
        
        for i, receiver_type in enumerate(celltypes[:2]):
            X_receiver = X[X.obs["celltype"] == receiver_type]
            X_receiver = X_receiver[np.argsort(-X_receiver.obsm["rc_C"][:, component])]
            
            df = expression_product_matrix(X_sender, X_receiver, ligand, receptor)
            df = average_product_matrix_ccc(df)
            
            sns.heatmap(df, ax=axes[i], cmap="viridis", vmax=0.12)
            axes[i].set_xlabel(f"Receiver {receiver_type}s")
            axes[i].set_ylabel(f"Sender {celltypes[0]}s")
            axes[i].set_title(f"{ligand}-{receptor}: {celltypes[0]}→{receiver_type}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_dir / "step11_celltype_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

print(f"\nAll figures saved to {output_dir}")
print("Done!")
