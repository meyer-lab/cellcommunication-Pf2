Scoring Methods (Ligand–Receptor Communication)
===============================================
 
This page explains how cell–cell communication scores are computed in the
project and what options are available.
 
What the code does by default
----------------------------
 
The core scoring functions are implemented under the `ccc` and `ccc_rise`
modules. By default the workflow uses an "expression product" score: for a
ligand gene A and receptor gene B, and expression vectors v (ligand across
senders) and w (receptor across receivers), the communication score is
computed as an outer product (v_i * w_j). This produces a sender×receiver
matrix for each ligand–receptor pair.
 
Alternative scoring options
---------------------------
 
The code also supports other simple combination rules:
 
- `expression_mean`: (v_i + w_j)/2 — the mean of ligand and receptor signals
- `expression_gmean`: geometric mean, sqrt(v_i * w_j)
 
Complexes and multi-subunit receptors
------------------------------------
 
Ligand or receptor names in the PPI/LR tables may encode complexes (e.g.
"ITGA1&ITGB1"). The helper functions `get_genes_from_complexes` and
`add_complexes_to_expression` parse these complex names using a separator
(default '&'), expand subunits, and compute an aggregate expression for the
complex (by default the minimum expression among subunits is used) when all
subunits are present in the expression matrix.
 
Custom scoring functions
------------------------
 
If you require a different scoring rule (for example, adding a threshold or
weighting receptor subunits differently) you can:
 
1. Compute the projected expression matrices yourself using RISE/parafac2,
   then
2. Call the communication-tensor construction functions with your custom
   matrices, or modify the `compute_ccc_matrix` function to implement the
   desired rule.
 
Notes and caveats
----------------
 
- Choice of scoring function changes absolute magnitudes and sparsity of the
  interaction tensor, which in turn affects downstream CP decomposition and
  rank selection.
- When using complex aggregation you must ensure all subunits are present in
  the expression matrix; otherwise the code will set complex expression to zero.
- We deliberately do not attempt to infer binding affinities or kinetics; the
  scores are simple expression-based proxies for potential signaling and should
  be interpreted with care.
 
If you are unsure which scoring rule is appropriate, leave a note here or in
the repository's docs TODO area and the lab lead can fill in recommended
options for common use cases.
