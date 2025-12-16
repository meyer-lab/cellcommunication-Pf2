Tutorial for CCC-RISE: Cell-Cell Communication Analysis
========================================================

This tutorial demonstrates the complete CCC-RISE workflow for analyzing cell-cell communication in single-cell RNA-seq data across experimental conditions.

Installation
------------

To add CCC-RISE to your Python package, add the following line to your ``requirements.txt`` and remake your virtual environment::

    git+https://github.com/meyer-lab/cellcommunication-Pf2.git@main

Preprocessing the Dataset
--------------------------

**Input Requirements**

Your AnnData object must meet the following requirements:

1. **Condition Index**: Include an observations column ``condition_unique_idxs`` that is a 0-indexed array indicating which condition each cell is derived from, along with the cell barcode. Condition 1 cells are indexed as 0, Condition 2 as 1, and so on.

2. **Preprocessing**: Your AnnData object must be preprocessed (doublets removed, genes filtered, normalized, and log-transformed) before running CCC-RISE. Standard preprocessing functions can assist with gene filtering, normalization, and assigning ``condition_unique_idxs``.

3. **Ligand-Receptor Pairs**: CCC-RISE requires a DataFrame of ligand-receptor pairs to analyze cell-cell communication. This can be obtained from resources like LIANA, CellPhoneDB, or other ligand-receptor databases.

   **Required DataFrame Format**:
   
   - Must contain columns named ``ligand`` and ``receptor``
   - Gene names should match those in your AnnData object (typically uppercase)
   - For protein complexes, subunits should be separated by ``&`` (e.g., ``CD74&CD44``)
   - Example structure::
   
       ligand     receptor
       CCL19      CCR7
       PTN        PTPRZ1
       CD74&CD44  CD44
   
   The package includes a default function ``import_ligand_receptor_pairs()`` that loads a curated database, but you can provide your own DataFrame following this format.
   
  

**Using prepare_dataset**

The ``prepare_dataset`` function assists with preprocessing your data. Parameters:

- ``X``: AnnData object containing raw count data in sparse matrix format
- ``condition_name``: Name of the column in ``X.obs`` that specifies experimental conditions for each cell
- ``geneThreshold``: Minimum mean expression threshold for gene filtering (genes with mean expression below this value are removed)
- ``deviance``: If True, applies deviance transformation instead of log normalization (default: False)

The function performs the following steps:

- Filters cells with fewer than 10 total counts
- Filters genes based on the ``geneThreshold`` parameter
- Normalizes total counts per cell to the median
- Scales gene expression values
- Applies log₁₀((1000 × normalized_value) + 1) transformation (or deviance transformation if specified)
- Creates ``condition_unique_idxs`` column in ``X.obs`` with 0-indexed condition assignments
- Pre-calculates gene means and stores in ``X.var["means"]``

**Import and Prepare the Dataset**

Import your dataset as an AnnData object with preprocessed data::

    from cellcommunicationpf2.import_data import prepare_dataset, import_ligand_receptor_pairs
    import anndata
    
    # Load your data
    X = anndata.read_h5ad("your_data.h5ad")
    
    # Prepare the dataset
    X = prepare_dataset(X, condition_name="condition", geneThreshold=0.01)
    
    # Load ligand-receptor pairs
    lr_pairs = import_ligand_receptor_pairs()

Choosing the Rank
------------------

**Assess Variance Explained by CCC-RISE**

Determine the optimal component/rank by plotting the variance explained (R²X) across different ranks. This helps balance model complexity with explanatory power::

    from cellcommunicationpf2.ccc_rise import run_fms_r2x_analysis
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    rank_list = list(range(5, 31, 5))
    results_df = run_fms_r2x_analysis(
        X, 
        lr_pairs=lr_pairs,
        rank_list=rank_list,
        runs=3
    )
    
    # Plot R2X
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=results_df, x='rank', y='r2x', ax=ax, marker='o')
    ax.set_xlabel('Rank')
    ax.set_ylabel('R²X (Variance Explained)')
    ax.set_title('CCC-RISE Model Performance')
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step1_r2x.png
   :width: 500px
   :align: center

**Evaluate Factor Stability with Factor Match Score (FMS)**

Measure the reproducibility of the CCC-RISE factorization across different ranks. An FMS above ~0.6 indicates stable components::

    # Plot FMS
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=results_df, x='rank', y='fms', ax=ax, marker='o')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Factor Match Score (FMS)')
    ax.set_title('CCC-RISE Component Stability')
    ax.axhline(y=0.6, color='r', linestyle='--', label='Stability Threshold')
    ax.legend()
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step2_fms.png
   :width: 500px
   :align: center

Running the Factorization
--------------------------

**Perform CCC-RISE Factorization**

Based on the variance explained and FMS, select a rank and perform the CCC-RISE factorization. This decomposes the cell-cell communication data into condition, sender cell, receiver cell, and ligand-receptor pair factors::

    from cellcommunicationpf2.ccc_rise import run_ccc_rise_workflow
    
    rank = 20
    X, r2x = run_ccc_rise_workflow(
        adata=X,
        rise_rank=rank,
        lr_pairs=lr_pairs,
        condition_column="condition_unique_idxs",
        doEmbedding=True,
        random_state=42
    )
    
    print(f"Variance Explained (R²X): {r2x:.4f}")

The function ``run_ccc_rise_workflow`` performs PARAFAC2 tensor decomposition on cell-cell communication scores. Parameters:

- ``adata``: AnnData object with preprocessed scRNA-seq data
- ``rise_rank``: Number of components to extract
- ``lr_pairs``: DataFrame of ligand-receptor pairs
- ``condition_column``: Column name in ``adata.obs`` containing condition indices
- ``doEmbedding``: If True, automatically computes PaCMAP embeddings (default: True)
- ``random_state``: Random seed for reproducibility (default: 42)

The output includes the original AnnData object with added results and the reconstruction error (R²X). The following are added to the AnnData object:

- **Weights**: ``X.uns["weights"]`` - The weights for each component
- **Condition Factors**: ``X.uns["A"]`` - Factors showing how each experimental condition contributes to each component
- **Sender Cell Factors**: ``X.uns["B"]`` - Factors showing sender cell patterns for each component
- **Receiver Cell Factors**: ``X.uns["C"]`` - Factors showing receiver cell patterns for each component
- **LR Pair Factors**: ``X.uns["D"]`` - Factors showing which ligand-receptor pairs are active in each component
- **Sender Projections**: ``X.obsm["rs_C"]`` - Cell projections as senders (matrix width equals the rank)
- **Receiver Projections**: ``X.obsm["rc_C"]`` - Cell projections as receivers (matrix width equals the rank)

Visualizing the Factors
------------------------

**Visualize Condition Factors**

Examine how each experimental condition contributes to the identified communication patterns::

    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    condition_factors = X.uns["A"]
    condition_names = X.obs["condition"].cat.categories
    
    sns.heatmap(
        condition_factors,
        ax=ax,
        cmap="viridis",
        yticklabels=condition_names,
        xticklabels=range(1, rank+1),
        cbar_kws={'label': 'Factor Loading'}
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Condition")
    ax.set_title("Condition Factors Heatmap")
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step3_condition_factors.png
   :width: 600px
   :align: center

**Condition Factors Heatmap**: This heatmap shows how each experimental condition (rows) contributes to each communication component (columns).

**Visualize Cell Embedding**

Explore the latent space of cells as senders or receivers using PaCMAP embeddings::

    from cellcommunicationpf2.utils import add_obs_cmp_label
    
    # Add labels for top cells in a specific component
    cmp = 5  # Component of interest
    X = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="sender")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sender embedding
    if "sender_embedding" in X.obsm:
        scatter = axes[0].scatter(
            X.obsm["sender_embedding"][:, 0],
            X.obsm["sender_embedding"][:, 1],
            c=X.obsm["rs_C"][:, cmp-1],
            cmap="viridis",
            s=1,
            alpha=0.5
        )
        axes[0].set_title(f"Sender Cells - Component {cmp}")
        plt.colorbar(scatter, ax=axes[0], label=f"Component {cmp} Loading")
    
    # Receiver embedding
    if "receiver_embedding" in X.obsm:
        scatter = axes[1].scatter(
            X.obsm["receiver_embedding"][:, 0],
            X.obsm["receiver_embedding"][:, 1],
            c=X.obsm["rc_C"][:, cmp-1],
            cmap="viridis",
            s=1,
            alpha=0.5
        )
        axes[1].set_title(f"Receiver Cells - Component {cmp}")
        plt.colorbar(scatter, ax=axes[1], label=f"Component {cmp} Loading")
    
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step4_cell_embeddings.png
   :width: 700px
   :align: center

**Visualize Ligand-Receptor Pair Factors**

Identify which ligand-receptor pairs are highly weighted in each component, revealing coordinated communication modules::

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
        xticklabels=range(1, rank+1),
        cbar_kws={'label': 'Factor Loading'}
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Ligand-Receptor Pair")
    ax.set_title("Top LR Pair Factors")
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step5_lr_pair_factors.png
   :width: 700px
   :align: center

**LR Pair Factors Heatmap**: This heatmap shows which ligand-receptor pairs (rows) are associated with each component (columns).

Interpreting a Component
-------------------------

**Investigate Cell Communication Patterns**

Overlay specific ligand-receptor expression products on the cell embedding to see which cell pairs drive communication::

    from cellcommunicationpf2.utils import expression_product_matrix
    
    # Select cells highly associated with a component
    cmp = 5
    X_labeled = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="sender")
    X_sender = X_labeled[X_labeled.obs["Label"] != "NoLabel"]
    
    X_labeled = add_obs_cmp_label(X, cmp=cmp, pos=True, top_perc=10, type="receiver")
    X_receiver = X_labeled[X_labeled.obs["Label"] != "NoLabel"]
    
    # Calculate expression product for a specific LR pair
    ligand = "CCL19"
    receptor = "CCR7"
    
    expr_product = expression_product_matrix(X_sender, X_receiver, ligand, receptor)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(expr_product, ax=ax, cmap="viridis", cbar_kws={'label': 'Expression Product'})
    ax.set_xlabel("Receiver Cells")
    ax.set_ylabel("Sender Cells")
    ax.set_title(f"{ligand}-{receptor} Communication")
    plt.tight_layout()
    plt.show()

.. image:: _static/tutorial_images/step6_communication_heatmap.png
   :width: 600px
   :align: center

**Communication Heatmap**: This visualization shows the communication scores between sender cells (rows) and receiver cells (columns) for a specific ligand-receptor pair.

**Analyze Top LR Pairs per Component**

Identify the most important ligand-receptor pairs for a specific component::

    cmp = 5
    lr_factor = X.uns["D"][:, cmp-1]
    lr_names = X.uns["lr_pair_names"]
    
    # Get top 10 LR pairs
    top_indices = np.argsort(np.abs(lr_factor))[-10:][::-1]
    
    print(f"\nTop 10 LR pairs for Component {cmp}:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {lr_names[idx]}: {lr_factor[idx]:.4f}")

.. image:: _static/tutorial_images/step7_top_lr_pairs.png
   :width: 600px
   :align: center
