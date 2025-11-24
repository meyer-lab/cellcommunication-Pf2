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

Practical suggestions
---------------------

- Use PaCMAP on a subset of cells if you have millions of cells — it scales
  better when run on representative subsets or pseudobulks.
- Tune `n_neighbors` depending on whether you want to emphasize local
  neighborhoods (smaller values) vs. global structure (larger values).
- If visualizing CP factor loadings for LR pairs, consider reducing to a
  per-pair score first (e.g., weight * factor_value) and running PaCMAP on the
  LR-pair matrix rather than per-cell data.

Example (conceptual)
---------------------

1. Extract per-cell projections from RISE: `B = adata.obsm['sc_B']`
2. Optionally subsample or pseudobulk to reduce noise.
3. Run PaCMAP on `B` with `n_components=2` and plot colored by condition or
   cell type.

Notes and TODOs
----------------

- TODO: include a short notebook demonstrating PaCMAP on the `BALF-COVID19` example data.
- The repository intentionally leaves PaCMAP out of the core workflow —
  embedding choices are downstream visualization steps and depend on user
  objectives.
