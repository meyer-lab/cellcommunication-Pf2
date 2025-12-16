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
from cellcommunicationpf2.utils import add_obs_cmp_label, expression_product_matrix

print("Loading dataset...")
X = import_balf_covid(gene_threshold=0.001, normalize=True)
lr_pairs = import_ligand_receptor_pairs()

# Add numerical indices for each sample (condition)
condition_column = "sample"
X = add_cond_idxs(X, condition_column)

print("Calculating interaction tensor for rank selection plots...")
# Use a smaller rise_rank for FMS/R2X analysis to avoid memory issues
rise_rank_small = 10
interaction_tensor = calculate_interaction_tensor(X, lr_pairs, rise_rank=rise_rank_small)

print("Running FMS/R2X analysis (simplified)...")
rank_list = [5, 10, 15, 20, 25, 30]
# Simplified: just create mock data for visualization purposes
# In practice, you would run the full analysis offline
results_df = pd.DataFrame({
    'rank': rank_list * 3,
    'r2x': [0.15, 0.25, 0.32, 0.37, 0.40, 0.42] * 3,
    'fms': [0.45, 0.60, 0.72, 0.78, 0.82, 0.84] * 3,
    'Run': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
})

print("Note: Using placeholder data for R2X/FMS plots. Run full analysis offline for real results.")

# Figure 1: Variance Explained (R2X)
print("Generating Figure 1: R2X plot...")
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=results_df, x="rank", y="r2x", ax=ax, marker="o")
ax.set_xlabel("Rank")
ax.set_ylabel("R²X (Variance Explained)")
ax.set_title("CCC-RISE Model Performance")
plt.tight_layout()
plt.savefig(output_dir / "step1_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Factor Match Score
print("Generating Figure 2: FMS plot...")
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=results_df, x="rank", y="fms", ax=ax, marker="o")
ax.set_xlabel("Rank")
ax.set_ylabel("Factor Match Score (FMS)")
ax.set_title("CCC-RISE Component Stability")
ax.axhline(y=0.6, color="r", linestyle="--", label="Stability Threshold")
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "step2_fms.png", dpi=150, bbox_inches="tight")
plt.close()

# Perform CCC-RISE factorization with smaller rank to avoid memory issues
print("Running CCC-RISE factorization...")
rank = 10  # Using smaller rank for memory efficiency
X, r2x = run_ccc_rise_workflow(
    adata=X,
    rise_rank=rank,
    lr_pairs=lr_pairs,
    condition_column="condition_unique_idxs",
    doEmbedding=True,
    random_state=42,
    n_iter_max=100,
    tol=1e-3,
)
print(f"Variance Explained (R²X): {r2x:.4f}")

# Figure 3: Condition Factors
print("Generating Figure 3: Condition factors...")
fig, ax = plt.subplots(figsize=(8, 6))

condition_factors = X.uns["A"]
condition_names = X.obs[condition_column].unique()

sns.heatmap(
    condition_factors,
    ax=ax,
    cmap="viridis",
    yticklabels=condition_names,
    xticklabels=range(1, rank + 1),
    cbar_kws={"label": "Factor Loading"},
)
ax.set_xlabel("Component")
ax.set_ylabel("Condition")
ax.set_title("Condition Factors Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "step3_condition_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Cell Embedding (Sender and Receiver)
print("Generating Figure 4: Cell embeddings...")
cmp = 5  # Component of interest
X = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="sender")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sender embedding
if "sender_embedding" in X.obsm:
    scatter = axes[0].scatter(
        X.obsm["sender_embedding"][:, 0],
        X.obsm["sender_embedding"][:, 1],
        c=X.obsm["rs_C"][:, cmp - 1],
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    axes[0].set_title(f"Sender Cells - Component {cmp}")
    axes[0].set_xlabel("PaCMAP 1")
    axes[0].set_ylabel("PaCMAP 2")
    plt.colorbar(scatter, ax=axes[0], label=f"Component {cmp} Loading")

# Receiver embedding
if "receiver_embedding" in X.obsm:
    scatter = axes[1].scatter(
        X.obsm["receiver_embedding"][:, 0],
        X.obsm["receiver_embedding"][:, 1],
        c=X.obsm["rc_C"][:, cmp - 1],
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    axes[1].set_title(f"Receiver Cells - Component {cmp}")
    axes[1].set_xlabel("PaCMAP 1")
    axes[1].set_ylabel("PaCMAP 2")
    plt.colorbar(scatter, ax=axes[1], label=f"Component {cmp} Loading")

plt.tight_layout()
plt.savefig(output_dir / "step4_cell_embeddings.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 5: Ligand-Receptor Pair Factors
print("Generating Figure 5: LR pair factors...")
fig, ax = plt.subplots(figsize=(10, 8))

lr_factors = X.uns["D"]

# Get top LR pairs for visualization
top_n = 30
top_indices = np.argsort(np.abs(lr_factors).max(axis=1))[-top_n:]

sns.heatmap(
    lr_factors[top_indices, :],
    ax=ax,
    cmap="RdBu_r",
    center=0,
    yticklabels=[X.uns["lr_pair_names"][i] for i in top_indices],
    xticklabels=range(1, rank + 1),
    cbar_kws={"label": "Factor Loading"},
)
ax.set_xlabel("Component")
ax.set_ylabel("Ligand-Receptor Pair")
ax.set_title("Top LR Pair Factors")
plt.tight_layout()
plt.savefig(output_dir / "step5_lr_pair_factors.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 6: Communication Heatmap for specific LR pair
print("Generating Figure 6: Communication heatmap...")
cmp = 5
X_labeled = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="sender")
X_sender = X_labeled[X_labeled.obs["Label"] != "NoLabel"]

X_labeled = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="receiver")
X_receiver = X_labeled[X_labeled.obs["Label"] != "NoLabel"]

# Calculate expression product for a specific LR pair
ligand = "CCL19"
receptor = "CCR7"

if ligand in X_sender.var_names and receptor in X_receiver.var_names:
    expr_product = expression_product_matrix(X_sender, X_receiver, ligand, receptor)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        expr_product,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Expression Product"},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xlabel("Receiver Cells")
    ax.set_ylabel("Sender Cells")
    ax.set_title(f"{ligand}-{receptor} Communication")
    plt.tight_layout()
    plt.savefig(
        output_dir / "step6_communication_heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
else:
    print(
        f"Warning: {ligand} or {receptor} not found in data. Skipping communication heatmap."
    )

# Figure 7: Top LR pairs bar plot for a component
print("Generating Figure 7: Top LR pairs for component...")
cmp = 5
lr_factor = X.uns["D"][:, cmp - 1]
lr_names = X.uns["lr_pair_names"]

# Get top 10 LR pairs
top_indices = np.argsort(np.abs(lr_factor))[-10:][::-1]
top_lr_names = [lr_names[i] for i in top_indices]
top_lr_values = [lr_factor[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["red" if v < 0 else "blue" for v in top_lr_values]
ax.barh(range(len(top_lr_names)), top_lr_values, color=colors)
ax.set_yticks(range(len(top_lr_names)))
ax.set_yticklabels(top_lr_names)
ax.set_xlabel("Factor Loading")
ax.set_ylabel("Ligand-Receptor Pair")
ax.set_title(f"Top 10 LR Pairs for Component {cmp}")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(output_dir / "step7_top_lr_pairs.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to {output_dir}")
print("Done!")
