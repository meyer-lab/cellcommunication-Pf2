.. _quickstart:

Quick Start Guide
=================

This tutorial demonstrates how to use CCC-RISE (Cell-Cell Communication via Reduction and Insight in Single-cell Exploration) to analyze cell-cell communication patterns in single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions. This guide uses COVID-19 patient BALF (bronchoalveolar lavage fluid) data as an example dataset.

Overview
--------

CCC-RISE identifies cell-cell communication patterns by:

1. Performing RISE dimensionality reduction on gene expression data across conditions
2. Computing communication scores between cell eigen-states (previously computed by RISE) using ligand-receptor pairs
3. Decomposing the resulting communication tensor using CP decomposition to identify key patterns

The workflow produces interpretable factors that reveal:

* **Condition patterns**: How communication changes across experimental conditions
* **Cell patterns**: Which sender and receiver cells drive communication
* **Ligand-receptor patterns**: Which signaling pathways are most important

Basic Workflow
--------------

Step 1: Import and Prepare Your Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use your own single-cell dataset or the provided example COVID-19 BALF dataset. For convenience, we provide a helper function to load a curated, preprocessed AnnData object:

.. code-block:: python

    import anndata
    from cellcommunicationpf2 import import_balf_covid

    # Load example single-cell data (will download if not present)
    adata = import_balf_covid(
        gene_threshold=0.01,   # Minimum mean expression (per gene)
        normalize=True         # Normalize and log1p-transform (recommended)
    )

**Required AnnData structure (if using your own data):**
- `adata.X`: (cells × genes) gene expression matrix (normalized and log-transformed recommended)
- `adata.obs`: must contain a column that uniquely identifies experimental conditions or samples
- `adata.var_names`: gene symbols matching those in your ligand-receptor table

**Ligand-receptor pairs:**
- You may use our provided ligand-receptor resource or import your own as a DataFrame with columns: 'ligand', 'receptor', and 'interaction_symbol'.

Please ensure your dataset is preprocessed for single-cell analysis (doublets removed, gene filtering, normalization, and log1p-transformation) for optimal results.


Step 2: Load and Inspect Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load the preprocessed COVID-19 BALF dataset and ligand-receptor pairs:

.. code-block:: python

    # Load single-cell data (automatically downloads if needed)
    adata = import_balf_covid(gene_threshold=0.01, normalize=True)
    
    # Load ligand-receptor pairs
    lr_pairs = import_ligand_receptor_pairs()
    
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Samples: {adata.obs['sample'].nunique()}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")


Step 3: Prepare Data for Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add numeric condition indices to enable tensor decomposition across samples:

.. code-block:: python

    # Add numeric indices for each sample

    adata = add_cond_idxs(adata, condition_key="sample")
    
    print(f"Condition indices: {adata.obs['condition_unique_idxs'].unique()}")

**Function: add_cond_idxs()**

This function creates numeric indices (0, 1, 2, ...) for each unique experimental condition/sample in your dataset. These indices are required for the RISE algorithm to correctly organize data across conditions.

**Parameters:**

* ``adata`` (AnnData): The single-cell dataset
* ``condition_key`` (str): Name of the column in ``adata.obs`` that defines experimental conditions

**What it does:**

The function maps each unique value in the specified column to a sequential number. For example:

* Sample "Control_1" → 0
* Sample "Control_2" → 1  
* Sample "Moderate_1" → 2
* Sample "Severe_1" → 3

These indices are stored in ``adata.obs['condition_unique_idxs']`` and are used internally by the RISE algorithm.

**Returns:**

The same AnnData object with an added column ``condition_unique_idxs`` in ``.obs``.

Step 4: Run CCC-RISE Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**How do I choose the rank values?** See :doc:`rank_selection` for a detailed explanation and practical advice on selecting both the RISE and CP decomposition ranks.

Perform the complete CCC-RISE workflow to identify communication patterns:

.. code-block:: python

    # Run CCC-RISE (returns updated adata with results and r2x value)
    adata, r2x = run_ccc_rise_workflow(
        adata=adata,
        rise_rank=35,           # RISE rank
        lr_pairs=lr_pairs,
        condition_column="sample",
        cp_rank=8,              # CP decomposition rank
        n_iter_max=1000,
        tol=1e-9,
        complex_sep='&'
    )
    
    print(f"Variance explained (R²X): {r2x:.3f}")

**Function: run_ccc_rise_workflow()**

This is the main function that executes the complete CCC-RISE pipeline. It performs RISE dimensionality reduction, computes cell-cell communication scores, and decomposes the communication tensor to identify interpretable patterns.

**Parameters:**

* ``adata`` (AnnData): The preprocessed single-cell dataset with gene expression data and cell metadata.

* ``rise_rank`` (int): The rank (number of components) for the RISE dimensionality reduction. This determines how many "eigen-states" or latent cell states are extracted from the data. Higher ranks capture more variation but may include noise. 
  
  In this example, ``rise_rank=35`` means we extract 35 cell eigen-states from the gene expression data.

* ``lr_pairs`` (DataFrame): The ligand-receptor interaction pairs loaded from ``import_ligand_receptor_pairs()``. This defines which gene pairs are considered for communication scoring.

* ``condition_column`` (str, default="sample"): The column name in ``adata.obs`` that defines experimental conditions. Must match the column used in ``add_cond_idxs()``.

* ``cp_rank`` (int, optional): The rank for the final CP (CANDECOMP/PARAFAC) tensor decomposition. If not specified, it defaults to the same value as ``rise_rank``. This determines how many communication "components" or patterns are identified. Lower values provide more interpretable results; higher values capture more complex patterns.

* ``n_iter_max`` (int, default=100): Maximum number of optimization iterations. The algorithm will stop either when it converges or reaches this limit. Higher values allow more time for convergence but take longer to run.

* ``tol`` (float, default=1e-9): Convergence tolerance. The algorithm stops when the change in the objective function is smaller than this value. Smaller values (e.g., 1e-9) ensure tighter convergence but may take longer.

* ``complex_sep`` (str, optional): Separator character for protein complexes in receptor names. Many receptors are multi-subunit complexes (e.g., "ITGA1&ITGB1" for integrin alpha-1/beta-1). The '&' character separates the subunits.
  
  - Use ``'&'`` if your ligand-receptor pairs include complexes
  - Use ``None`` to ignore complexes
  
  When specified, the function computes complex expression as the minimum expression of all subunits.

**What this function does:**

1. Performs RISE decomposition on the scRNA-seq data
2. RISE projects cells into cell eigen-states using the projection matrices solved by RISE
3. Computes communication scores between all sender-receiver cell state pairs for each ligand-receptor (LR) pair
4. Creates a 4D interaction tensor: (conditions × sender eigen-states × receiver eigen-states × LR pairs)
5. Decomposes this 4D interaction tensor using CPD
6. Stores all results in the AnnData object for downstream analysis

**Returns:**

A tuple of two values:

1. ``adata`` (AnnData): The input object updated with results stored in:

   - ``.uns["A"]``: Condition factor matrix
   - ``.uns["B"]``: Sender eigen-state factor matrix
   - ``.uns["C"]``: Receiver eigen-state factor matrix
   - ``.uns["D"]``: Ligand-receptor pair factor matrix
   - ``.uns["weights"]``: Component importance weights
   - ``.uns["r2x"]``: Variance explained by the model
   - ``.uns["lr_pairs"]``: Names of included ligand-receptor pairs
   - ``.obsm["projections"]``: Cell projections based on RISE  
   - ``.obsm["sc_B"]``: Cell projections weighted by sender cell state factors (each cell × n_components)
   - ``.obsm["rc_C"]``: Cell projections weighted by receiver cell state factors (each cell × n_components)

2. ``r2x`` (float): The proportion of variance explained by the model (0 to 1). Higher values indicate the model captures more of the communication patterns in the data.

Step 5: Visualize and Interpret Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The best way to interpret your CCC-RISE results is by using the visualization helpers provided in the `cellcommunicationpf2.figures` module. These functions are used in our main and supplementary figures (e.g., see `figures/figureA2b.py`) and provide clean, publication-ready plots of all major outputs with minimal code.

**Example: Recreating Figure A2b—visualizing all factor matrices**

.. code-block:: python

    import matplotlib.pyplot as plt
    from cellcommunicationpf2.figures.commonFuncs.plotFactors import (
        plot_condition_factors,
        plot_eigenstate_factors,
        plot_lr_factors,
    )
    from cellcommunicationpf2.figures.common import getSetup, subplotLabel
    from cellcommunicationpf2.figures.commonFuncs.plotGeneral import rotate_yaxis

    # Setup: 4 horizontal subplots for the 4 factor matrices (as in Figure A2b)
    axs, fig = getSetup((20, 8), (1, 4))
    subplotLabel(axs)

    # You may want to cluster or annotate by group (see figureA2b for further customization)
    # For this example, we'll just use default settings

    # Patient/sample condition factors
    plot_condition_factors(
        adata, 
        axs[0],
        cond="sample",
        normalize=True,
    )
    axs[0].set_title("Condition Factors")

    # Sender cell eigen-states
    plot_eigenstate_factors(
        adata,
        axs[1],
        factor_type="B"
    )
    axs[1].set_title("Sender Cell Eigen-states")
    rotate_yaxis(axs[1], rotation=0)

    # Receiver cell eigen-states
    plot_eigenstate_factors(
        adata, 
        axs[2],
        factor_type="C"
    )
    axs[2].set_title("Receiver Cell Eigen-states")
    rotate_yaxis(axs[2], rotation=0)

    # Ligand-Receptor pair factors (sorted & trimmed for readability)
    plot_lr_factors(
        adata, 
        axs[3], 
        trim=True, 
        weight=0.06
    )
    axs[3].set_title("Ligand-Receptor Pairs")

    plt.tight_layout()
    plt.show()

**What do these plots show?**

- **Condition Factors:** Samples grouped/annotated by patient or experimental label, revealing shifts in communication patterns across conditions.
- **Sender/Receiver Eigen-states:** Heatmaps showing which cell states (identified by RISE) act as strong senders or receivers in each communication program/component.
- **Ligand-Receptor Factors:** The LR pairs that drive each program (optionally sorted and trimmed for clarity).

All these high-level helpers handle normalization, annotation, and color scales for you—see `figures/figureA2b.py` for full details and advanced grouping/formatting options.

If you want to do further/custom analysis or plotting, you can always access the raw factor arrays in the AnnData object:

.. code-block:: python

    condition_factors = adata.uns["A"]      # (n_samples, n_components)
    sender_factors = adata.uns["B"]         # (rise_rank, n_components)
    receiver_factors = adata.uns["C"]       # (rise_rank, n_components)
    lr_factors = adata.uns["D"]             # (n_lr_pairs, n_components)
    weights = adata.uns["weights"]          # (n_components,)

    print("Condition factors shape:", condition_factors.shape)
    print("Component weights:", weights)
    print("Number of components identified:", len(weights))

Next Steps
----------

* Explore :doc:`api/index` for complete function documentation
