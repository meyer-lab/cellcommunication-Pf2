Choosing Ranks for CCC-RISE
===========================

This page explains how to choose the two main ranks used in the CCC‑RISE workflow:

- The RISE / PARAFAC2 rank (number of latent cell eigen-states)
- The CP rank (number of communication components / factors)

Overview
--------

Choosing a rank is somewhat nuanced and we don't have a one-size-fits-all rule. The codebase provides two metrics
to guide this choice:

- R²X (R2X): the fraction of variance in the interaction tensor explained by a
  CP decomposition. Higher R²X generally indicates better reconstruction.
- FMS (Factor Match Score): a stability metric that compares factor solutions
  from bootstrap/resampled data to a reference decomposition. Higher FMS means
  more stable, reproducible components.

Key ideas
---------

- R²X tends to increase with CP rank: adding components lets the model explain
  more variance (but risks overfitting and producing components with little
  biological meaning).
- FMS tends to decrease as CP rank increases because more components are
  harder to estimate stably; bootstrapped runs will often place small
  components in different modes.

Rules of thumb
--------------

The project team has found the following workflow helpful when selecting ranks:

1. Pick a range of candidate RISE ranks (e.g. test multiple values). For each
   RISE rank, compute the projected expression matrices and the interaction
   tensor.
2. For each candidate CP rank, compute CP decomposition on the interaction
   tensor and record R²X and FMS via bootstrapping.
3. Inspect the trade-off: look for an elbow in R²X growth and a region where
   FMS remains reasonably high.

Suggested placeholders
----------------------

- Typical RISE rank candidates: TODO: provide a recommended numeric range (e.g. 10–50).
- Typical CP ranks to test: TODO: provide a recommended numeric range (e.g. 2–12).

Leave these placeholders for lab-specific recommendations.

Picking individual components with R²X
-------------------------------------

R²X measures variance explained by the whole CP decomposition. It is not a
component-level p-value. If you want to know whether a particular component is
well supported, combine R²X trends with:

- the magnitude of the component weight (larger weights explain more variance),
- stability across bootstraps (component appears consistently and has high FMS),
- biological interpretability (do the top cell states and ligand-receptor pairs
  make sense together?).

When results are surprising
--------------------------

If you get unexpectedly low FMS or components that look biologically implausible:

- Re-check preprocessing (normalization, gene filtering) — small changes here
  can strongly affect results. See the preprocessing page.
- Increase the number of bootstrap runs for FMS to get a more stable estimate.
- Consider lowering the CP rank and inspecting the larger components first.
- If you still think the methods or results are wrong, open an issue in the
  repository and attach a minimal reproducible example.

Further reading
---------------

See :doc:`preprocessing` and :doc:`scoring_methods` for more details that affect
rank behavior.
