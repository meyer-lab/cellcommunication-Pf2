.. _quickstart:

Quick Start Guide
=================

This guide walks you through a basic CCC-RISE analysis using the COVID-19 BALF dataset.

Basic Workflow
--------------

Step 1: Import Required Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import anndata
    from cellcommunicationpf2 import (
        import_balf_covid, 
        import_ligand_receptor_pairs,
    )
    from cellcommunicationpf2.tensor import run_ccc_rise_workflow
    from cellcommunicationpf2.import_data import add_cond_idxs

Step 2: Load Data
^^^^^^^^^^^^^^^^^

Load the preprocessed COVID-19 BALF dataset:

.. code-block:: python

    # Load single-cell data (automatically downloads if needed)
    adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    
    # Load ligand-receptor pairs
    lr_pairs = import_ligand_receptor_pairs()
    
    print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Samples: {adata.obs['sample'].nunique()}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")

Step 3: Run CCC-RISE Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform the complete CCC-RISE workflow:

.. code-block:: python

    # Add numeric indices for each sample
    adata = add_cond_idxs(adata, condition_column="sample")
    
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

Step 4: Access Results
^^^^^^^^^^^^^^^^^^^^^^^

Extract and examine the decomposition factors:

.. code-block:: python

    # Get factor matrices
    condition_factors = adata.uns["A"]     # (n_samples, n_components)
    sender_factors = adata.uns["B"]        # (rise_rank, n_components)
    receiver_factors = adata.uns["C"]      # (rise_rank, n_components)
    lr_factors = adata.uns["D"]            # (n_lr_pairs, n_components)
    weights = adata.uns["weights"]         # (n_components,)
    
    print(f"Condition factors shape: {condition_factors.shape}")
    print(f"Component weights: {weights}")

Step 5: Visualize Results
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create basic visualizations:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot component weights
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(weights)), weights)
    ax.set_xlabel('Component')
    ax.set_ylabel('Weight')
    ax.set_title('Component Importance')
    plt.show()
    
    # Heatmap of condition factors
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(condition_factors, cmap='RdBu_r', center=0, ax=ax)
    ax.set_xlabel('Component')
    ax.set_ylabel('Sample')
    ax.set_title('Sample-Component Associations')
    plt.tight_layout()
    plt.show()

Complete Example Script
------------------------

Here's a complete working example:

.. code-block:: python

    """Complete CCC-RISE analysis example"""
    
    from cellcommunicationpf2 import (
        import_balf_covid,
        import_ligand_receptor_pairs,
    )
    from cellcommunicationpf2.import_data import add_cond_idxs
    from cellcommunicationpf2.tensor import run_ccc_rise_workflow
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load data
    print("Loading data...")
    adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()
    adata = add_cond_idxs(adata, condition_column="sample")
    
    # Run analysis
    print("Running CCC-RISE...")
    adata, r2x = run_ccc_rise_workflow(
        adata=adata,
        rise_rank=35,
        lr_pairs=lr_pairs,
        condition_column="sample",
        cp_rank=8,
        n_iter_max=1000,
        tol=1e-9
    )
    
    print(f"\nAnalysis complete!")
    print(f"R²X: {r2x:.3f}")
    print(f"Identified {adata.uns['A'].shape[1]} components")
    
    # Visualize
    print("Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Component weights
    axes[0].bar(range(len(adata.uns["weights"])), 
                adata.uns["weights"])
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Weight')
    axes[0].set_title('Component Importance')
    
    # Condition factors heatmap
    sns.heatmap(adata.uns["A"], cmap='RdBu_r', center=0, 
                ax=axes[1], cbar_kws={'label': 'Factor Loading'})
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Sample')
    axes[1].set_title('Sample-Component Matrix')
    
    plt.tight_layout()
    plt.savefig('ccc_rise_results.png', dpi=300, bbox_inches='tight')
    print("Saved results to ccc_rise_results.png")

Next Steps
----------

* Explore :doc:`api/index` for complete function documentation
