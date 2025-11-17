.. _quickstart:

Quick Start Guide
=================

This tutorial demonstrates how to use CCC-RISE (Cell-Cell Communication via Reduction and Insight in Single-cell Exploration) to analyze cell-cell communication patterns in single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions. This guide uses COVID-19 patient BALF (bronchoalveolar lavage fluid) data as an example dataset.

Overview
--------

CCC-RISE identifies cell-cell communication patterns by:

1. Performing RISE dimensionality reduction on gene expression data across conditions
2. Computing communication scores between cell types using ligand-receptor pairs
3. Decomposing the resulting communication tensor to identify key patterns

The workflow produces interpretable factors that reveal:

* **Condition patterns**: How communication changes across experimental conditions
* **Cell type patterns**: Which sender and receiver cell types drive communication
* **Ligand-receptor patterns**: Which signaling pathways are most important

Basic Workflow
--------------

Step 1: Import Required Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, import the necessary functions from the cellcommunicationpf2 package:

.. code-block:: python

    import anndata
    from cellcommunicationpf2 import (
        import_balf_covid, 
        import_ligand_receptor_pairs,
    )
    from cellcommunicationpf2.tensor import run_ccc_rise_workflow
    from cellcommunicationpf2.import_data import add_cond_idxs

**What these modules do:**

* ``anndata``: Provides the AnnData structure for storing single-cell data
* ``import_balf_covid``: Loads the example COVID-19 BALF dataset
* ``import_ligand_receptor_pairs``: Loads curated ligand-receptor interaction pairs
* ``run_ccc_rise_workflow``: Main function that performs the complete CCC-RISE analysis
* ``add_cond_idxs``: Helper function to add numeric indices for experimental conditions

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

**Function: import_balf_covid()**

This function loads bronchoalveolar lavage fluid (BALF) immune cell data from COVID-19 patients. The dataset contains ~63,000 cells from 3 control patients, 3 moderate COVID-19 patients, and 6 severe COVID-19 patients.

**Parameters:**

* ``gene_threshold`` (float, default=0.01): Minimum mean expression level for a gene to be included in the analysis. Genes with average expression below this threshold across all cells are filtered out. Lower values (e.g., 0.001) retain more genes; higher values (e.g., 0.01) perform more aggressive filtering. This removes lowly expressed genes that add noise to the analysis.

* ``normalize`` (bool, default=True): Whether to normalize the expression data. When True, the function:
  
  - Normalizes library sizes (total counts per cell) to account for technical variation
  - Scales gene expression to make values comparable across cells
  - Log-transforms the data to stabilize variance
  
  Normalization is recommended for most analyses to ensure cells with different sequencing depths are comparable.

**Returns:**

An AnnData object containing:

* ``.X``: Normalized gene expression matrix (cells × genes)
* ``.obs``: Cell metadata including sample IDs, cell types, patient severity
* ``.var``: Gene metadata including gene names
* ``.obs['condition_unique_idxs']``: Numeric indices (0, 1, 2, ...) for each sample/condition

**Function: import_ligand_receptor_pairs()**

This function loads a curated database of known ligand-receptor interactions from the literature. These pairs define which genes encode ligands (signaling molecules) and which encode their corresponding receptors.

**Returns:**

A pandas DataFrame with columns:

* ``ligand``: Gene name of the signaling ligand
* ``receptor``: Gene name(s) of the receptor (may include protein complexes like "ITGA1&ITGB1")
* ``interaction_symbol``: Combined name for the interaction pair

The function loads ~2,000 validated human ligand-receptor pairs used to compute communication scores.

Note: the implementation of `import_ligand_receptor_pairs` reads a zstd-compressed CSV by default (filename set in the code). When `update_interaction_names=True` (the default) the function will try to populate/normalize `ligand`, `receptor`, and `interaction_symbol` columns from available fields (for example `interaction_name_2`), and will upper-case / replace some characters ("+" → "&"). If you supply your own CSV/DataFrame, ensure it contains at least `ligand` and `receptor`, and optionally `interaction_symbol`.

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

* ``cp_rank`` (int, optional): The rank for the final CP (CANDECOMP/PARAFAC) tensor decomposition. This determines how many communication "components" or patterns are identified. Lower values provide more interpretable results; higher values capture more complex patterns.

* ``n_iter_max`` (int, default=100): Maximum number of optimization iterations. The algorithm will stop either when it converges or reaches this limit. Higher values allow more time for convergence but take longer to run.

* ``tol`` (float, default=1e-9): Convergence tolerance. The algorithm stops when the change in the objective function is smaller than this value. Smaller values (e.g., 1e-9) ensure tighter convergence but may take longer.

* ``complex_sep`` (str, optional): Separator character for protein complexes in receptor names. Many receptors are multi-subunit complexes (e.g., "ITGA1&ITGB1" for integrin alpha-1/beta-1). The '&' character separates the subunits.
  
  - Use ``'&'`` if your ligand-receptor pairs include complexes
  - Use ``None`` to ignore complexes
  
  When specified, the function computes complex expression as the minimum expression of all subunits.

**What this function does:**

1. Performs RISE decomposition on gene expression data separately for each condition
2. Projects cells onto the RISE factors to get low-dimensional representations
3. Computes communication scores between all sender-receiver cell state pairs for each ligand-receptor pair
4. Creates a 4D interaction tensor: (conditions × sender states × receiver states × LR pairs)
5. Decomposes this tensor using CP decomposition to identify major communication patterns
6. Standardizes factors for interpretability (scales and orients them consistently)
7. Stores all results in the AnnData object for downstream analysis

**Returns:**

A tuple of two values:

1. ``adata`` (AnnData): The input object updated with results stored in:
   
   - ``.uns["A"]``: Condition factor matrix
   - ``.uns["B"]``: Sender cell state factor matrix  
   - ``.uns["C"]``: Receiver cell state factor matrix
   - ``.uns["D"]``: Ligand-receptor pair factor matrix
   - ``.uns["weights"]``: Component importance weights
   - ``.uns["r2x"]``: Variance explained by the model
   - ``.uns["lr_pairs"]``: Names of included ligand-receptor pairs
   - ``.obsm["projections"]``: Cell projections onto RISE factors
   - ``.obsm["sc_B"]``: Sender cell embeddings
   - ``.obsm["rc_C"]``: Receiver cell embeddings

2. ``r2x`` (float): The proportion of variance explained by the model (0 to 1). Higher values indicate the model captures more of the communication patterns in the data.

Step 5: Access and Interpret Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract and examine the decomposition factors to understand communication patterns:

.. code-block:: python

    # Get factor matrices
    condition_factors = adata.uns["A"]     # (n_samples, n_components)
    sender_factors = adata.uns["B"]        # (rise_rank, n_components)
    receiver_factors = adata.uns["C"]      # (rise_rank, n_components)
    lr_factors = adata.uns["D"]            # (n_lr_pairs, n_components)
    weights = adata.uns["weights"]         # (n_components,)
    
    print(f"Condition factors shape: {condition_factors.shape}")
    print(f"Component weights: {weights}")
    print(f"Number of components identified: {len(weights)}")

**Understanding the Factor Matrices:**

The CCC-RISE decomposition breaks down the complex 4D communication tensor into four interpretable factor matrices. Each factor matrix reveals a different aspect of the communication patterns:

**1. Condition Factors (Factor A):**

* **Shape**: (n_samples, n_components)
* **Interpretation**: Each row represents one experimental condition/sample (e.g., control, moderate COVID, severe COVID). Each column represents one communication component.
* **Values**: Positive values indicate the component is more active in that condition; negative values indicate it is suppressed. The magnitude indicates strength.
* **Example**: If component 1 has high positive values for severe COVID samples and negative values for controls, it represents communication patterns enriched in severe disease.

**2. Sender Cell State Factors (Factor B):**

* **Shape**: (rise_rank, n_components)  
* **Interpretation**: Each row represents one of the latent cell states identified by RISE (these are combinations of cell types with similar expression). Each column represents one communication component.
* **Values**: Positive values indicate the cell state acts as an important sender (produces ligands) in that component; negative values indicate it does not.
* **Example**: If component 1 has high values for cell state 5, then cells in state 5 are key signal senders for that communication pattern.

**3. Receiver Cell State Factors (Factor C):**

* **Shape**: (rise_rank, n_components)
* **Interpretation**: Each row represents one latent cell state. Each column represents one communication component.
* **Values**: Positive values indicate the cell state acts as an important receiver (expresses receptors) in that component; negative values indicate it does not.
* **Example**: If component 1 has high values for cell state 12, then cells in state 12 are key signal receivers for that communication pattern.

**4. Ligand-Receptor Factors (Factor D):**

* **Shape**: (n_lr_pairs, n_components)
* **Interpretation**: Each row represents one ligand-receptor interaction pair. Each column represents one communication component.
* **Values**: Positive values indicate the LR pair is important in that component; negative values indicate it is not.
* **Example**: If component 1 has high values for "IL6-IL6R" and "TNF-TNFRSF1A", these inflammatory signaling pathways drive that communication pattern.

**5. Component Weights:**

* **Shape**: (n_components,)
* **Interpretation**: The relative importance of each component. Components with larger weights explain more variance in the communication data.
* **Values**: Always positive. Components are ordered by condition variance (Gini coefficient) rather than by weight magnitude.
* **Example**: If weights = [2.5, 1.8, 1.2, ...], these indicate the relative importance of each identified pattern.

**How to Interpret Components Together:**

Each component tells a coordinated story by combining information from all four factors:

* **Component 1** might represent: "In severe COVID (high in Factor A), macrophage-like cell states (high sender in Factor B) signal to T cell-like states (high receiver in Factor C) via IL6-IL6R and TNF-TNFRSF1A pathways (high in Factor D)"
* **Component 2** might represent: "In control samples (high in Factor A), epithelial-like states signal to immune cell states via homeostatic pathways"

The model identifies these patterns automatically without prior knowledge of the biology.

Advanced Usage Tips
-------------------

Choosing RISE Rank
^^^^^^^^^^^^^^^^^^

The RISE rank determines how many latent cell states are extracted. To choose an appropriate rank:

1. **Plot variance explained (R²X) vs. rank**: Run RISE with different ranks and plot R²X values. Look for an "elbow" where R²X plateaus.

2. **Use Factor Match Score (FMS)**: Compute FMS between factorizations of the original data and bootstrapped versions. Higher FMS indicates more stable components.

3. **Biological interpretability**: Choose a rank where the latent cell states correspond to meaningful biological populations.

Example code to test multiple ranks:

.. code-block:: python

    ranks = [10, 20, 30, 40, 50]
    r2x_values = []
    
    for rank in ranks:
        adata_test, r2x = run_ccc_rise_workflow(
            adata=adata.copy(),
            rise_rank=rank,
            lr_pairs=lr_pairs,
            condition_column="sample",
            cp_rank=8,
            n_iter_max=500,
            tol=1e-6
        )
        r2x_values.append(r2x)
        print(f"Rank {rank}: R²X = {r2x:.3f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, r2x_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('RISE Rank')
    plt.ylabel('Variance Explained (R²X)')
    plt.title('Model Selection: RISE Rank vs. R²X')
    plt.grid(True, alpha=0.3)
    plt.show()

Choosing CP Rank
^^^^^^^^^^^^^^^^

The CP rank determines how many communication components are identified. To choose an appropriate rank:

1. **Biological interpretability**: Choose a rank where most components have clear biological interpretations.

2. **Check component weights**: If many components have very small weights relative to others, you may be using too many components.

3. **Use cross-validation**: Hold out samples and test how well the model predicts their communication patterns.

Working with Your Own Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To analyze your own single-cell dataset:

1. **Prepare your AnnData object** with:
   
   - ``.X``: Gene expression matrix (cells × genes), preferably sparse
   - ``.obs``: Cell metadata with a column defining experimental conditions
   - ``.var_names``: Gene symbols (must match ligand-receptor pair database)

2. **Preprocess your data**:

   .. code-block:: python
   
       from cellcommunicationpf2.import_data import prepare_dataset
       
       # Load your data
       adata = anndata.read_h5ad("your_data.h5ad")
       
       # Preprocess (filter genes, normalize)
       adata = prepare_dataset(
           adata,
           condition_name="your_condition_column",  # e.g., "sample" or "timepoint"
           geneThreshold=0.01,                      # Filter low-expression genes
           normalize=True                           # Normalize and log-transform
       )

3. **Run CCC-RISE workflow** as shown in the examples above.

4. **Use your own ligand-receptor pairs** (optional):

   .. code-block:: python
   
       import pandas as pd
       
       # Load custom LR pairs
       custom_lr = pd.DataFrame({
           'ligand': ['TGFB1', 'IL6', 'TNF'],
           'receptor': ['TGFBR1&TGFBR2', 'IL6R&IL6ST', 'TNFRSF1A'],
           'interaction_symbol': ['TGFB1-TGFBR1&TGFBR2', 'IL6-IL6R&IL6ST', 'TNF-TNFRSF1A']
       })
       
       # Run with custom pairs
       adata, r2x = run_ccc_rise_workflow(
           adata=adata,
           rise_rank=30,
           lr_pairs=custom_lr,  # Use custom pairs instead
           condition_column="sample",
           cp_rank=8
       )

Next Steps
----------

* Explore :doc:`api/index` for complete function documentation
