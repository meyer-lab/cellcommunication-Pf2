Choosing Ranks for CCC-RISE
===========================

This page explains how to choose the two main ranks used in the CCC‑RISE workflow:

- The RISE / PARAFAC2 rank (number of latent cell eigen-states)
- The CP rank (number of communication components / factors)

Key idea
---------

- R²X tends to increase with CP rank: adding components lets the model explain
  more variance (but risks overfitting and producing components with little
  biological meaning).

Rules of thumb
--------------

We have found the following workflow helpful when selecting ranks:

1. Pick a range of candidate RISE ranks (e.g. test multiple values). For each
   RISE rank, compute the projected expression matrices and the interaction
   tensor.
2. For each candidate CP rank, compute CP decomposition on the interaction
   tensor and record R²X and FMS via bootstrapping.
3. Inspect the trade-off: look for an elbow in R²X growth and a region where
   FMS remains reasonably high.

Example: Finding Optimal Ranks
--------------------------------

The following example demonstrates how to systematically test different rank combinations
to find optimal parameters for your dataset.

Step 1: Test RISE Ranks
^^^^^^^^^^^^^^^^^^^^^^^

First, test different RISE ranks to find a stable decomposition. This evaluates the
stability of the PARAFAC2 decomposition across different ranks:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cellcommunicationpf2.import_data import (
        import_balf_covid,
        add_cond_idxs,
    )
    from cellcommunicationpf2.figures.commonFuncs.plotGeneral import plot_fms_r2x_diff_ranks

    # Load and prepare data
    adata = import_balf_covid(gene_threshold=0.01, normalize=True)
    adata = add_cond_idxs(adata, condition_key="sample")

    # Test a range of RISE ranks
    rise_ranks = list(range(5, 41, 5))  # [5, 10, 15, 20, 25, 30, 35, 40]
    runs = 3  # Number of bootstrap runs for stability assessment

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot FMS and R2X across RISE ranks
    plot_fms_r2x_diff_ranks(
        adata,
        condition_name="sample",
        ax1=ax1,
        ax2=ax2,
        ranksList=rise_ranks,
        runs=runs
    )

    ax1.set_title("FMS vs RISE Rank")
    ax1.set_xlabel("RISE Rank")
    ax1.set_ylabel("Factor Match Score")
    ax1.set_ylim(0, 1)

    ax2.set_title("R²X vs RISE Rank")
    ax2.set_xlabel("RISE Rank")
    ax2.set_ylabel("R²X (Variance Explained)")

    plt.tight_layout()
    plt.show()

Look for RISE ranks where FMS remains high (typically > 0.7) and R²X shows
reasonable variance explained. Common choices are in the range 10–50 depending
on dataset size.

Step 2: Test CP Ranks
^^^^^^^^^^^^^^^^^^^^^

Once you've selected a RISE rank, test different CP ranks on the resulting
interaction tensor:

.. code-block:: python

    import seaborn as sns
    import matplotlib.pyplot as plt
    from cellcommunicationpf2.import_data import import_ligand_receptor_pairs
    from cellcommunicationpf2.tensor import (
        calculate_interaction_tensor,
        run_fms_r2x_analysis,
    )

    # Load ligand-receptor pairs
    lr_pairs = import_ligand_receptor_pairs()

    # Select a RISE rank based on Step 1 results
    selected_rise_rank = 35  # Example: choose based on Step 1 analysis

    # Calculate interaction tensor using the selected RISE rank
    print("Computing interaction tensor...")
    interaction_tensor = calculate_interaction_tensor(
        adata,
        lr_pairs,
        rise_rank=selected_rise_rank
    )
    print(f"Interaction tensor shape: {interaction_tensor.shape}")

    # Test a range of CP ranks
    cp_ranks = list(range(2, 15, 2))  # [2, 4, 6, 8, 10, 12, 14]
    runs = 3  # Number of bootstrap runs

    # Run FMS and R2X analysis
    print("Testing CP ranks...")
    results_df = run_fms_r2x_analysis(
        interaction_tensor,
        rank_list=cp_ranks,
        runs=runs,
        svd_init="svd"
    )

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=results_df, x="Component", y="FMS", ax=ax1, label="FMS")
    ax1.set_title("FMS vs CP Rank")
    ax1.set_xlabel("CP Rank")
    ax1.set_ylabel("Factor Match Score")
    ax1.set_ylim(0, 1)
    ax1.legend()

    sns.lineplot(data=results_df, x="Component", y="R2X", ax=ax2, color="orange", label="R²X")
    ax2.set_title("R²X vs CP Rank")
    ax2.set_xlabel("CP Rank")
    ax2.set_ylabel("R²X (Variance Explained)")
    ax2.set_ylim(0, results_df["R2X"].max() + 0.02)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary by CP Rank:")
    summary = results_df.groupby("Component")[["FMS", "R2X"]].mean()
    print(summary)

Step 3: Select Optimal Ranks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the plots:

1. **RISE rank**: Choose a rank where FMS is high (typically > 0.7) and R²X
   shows good variance explained. Avoid ranks where FMS drops significantly.

2. **CP rank**: Look for an "elbow" in the R²X curve where additional components
   provide diminishing returns. Choose a rank where:
   - R²X shows good variance explained without overfitting
   - The rank is interpretable (not too many components to analyze)

Example interpretation:
- If R²X plateaus around rank 8 and FMS drops below 0.6 after rank 10,
  a CP rank of 8–10 might be optimal.
- If FMS remains high (> 0.7) up to rank 12, you can safely use higher ranks
  for more detailed analysis.

Typical Rank Ranges
-------------------

Based on common usage patterns:

- **RISE rank candidates**: Typically 10–50, depending on dataset size and complexity.
  Smaller datasets (fewer cells/conditions) may use 10–25, while larger datasets
  may benefit from 25–50.

- **CP ranks to test**: Typically 2–15. Higher ranks provide more granular
  patterns but may be harder to interpret.

These ranges are starting points; adjust based on your specific dataset characteristics
and the stability metrics (FMS and R²X) you observe.

