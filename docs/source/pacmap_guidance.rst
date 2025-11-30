PaCMAP guidance and usage notes
================================

This short page describes how PaCMAP can be used in exploratory analysis
around CCC‑RISE outputs. It is guidance only — the codebase does not include
an automated PaCMAP step by default.

Why use PaCMAP here?
---------------------

PaCMAP is a fast dimensionality reduction method that preserves both local and
global structure. You can use it to visualize cell embeddings (e.g., the
per-condition projections `X.obsm['sc_B']` and `X.obsm['rc_C']` produced by
RISE) or to visualize factor scores from the CP decomposition.

Example: Visualizing cells with PaCMAP
---------------------------------------

The following example is adapted from `figureA2c_d.py`, which shows PaCMAP visualizations
of CCC-RISE results on BALF COVID-19 data. The figure displays cells colored by cell type
and disease condition to explore how different cell populations and experimental conditions
are distributed in the PaCMAP embedding space.

After running `run_ccc_rise_workflow` with `doEmbedding=True` (the default), the
PaCMAP coordinates are automatically computed and stored in `adata.obsm["PaCMAP"]`.
You can then visualize cells colored by metadata labels:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cellcommunicationpf2.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
    import seaborn as sns

    # After running run_ccc_rise_workflow, PaCMAP coordinates are in adata.obsm["PaCMAP"]
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Color by cell type
    palette = sns.color_palette("Set3").as_hex()
    plot_labels_pacmap(adata, labelType="cell_type", ax=axes[0], color_key=palette)
    axes[0].set_title("Cells colored by cell type")

    # Color by condition
    condition_palette = sns.color_palette("Set2", n_colors=3).as_hex()
    plot_labels_pacmap(adata, labelType="condition", ax=axes[1], color_key=condition_palette)
    axes[1].set_title("Cells colored by condition")

    # Color by sample (uses default colormap)
    plot_labels_pacmap(adata, labelType="sample", ax=axes[2])
    axes[2].set_title("Cells colored by sample")

    plt.tight_layout()
    plt.show()


Note: The repository intentionally leaves PaCMAP out of the core workflow —
embedding choices are downstream visualization steps and depend on user objectives.
