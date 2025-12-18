.. CCC-RISE documentation master file

Welcome to CCC-RISE's documentation!
====================================

Overview
--------

CCC-RISE (Cell-Cell Communication with Reduction and Insight in Single-cell Exploration) is an unsupervised, tensor-based computational method designed for the integrative analysis of cell-cell communication in single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions, such as drug treatments, patient cohorts, or time points. Built upon the CP (CANDECOMP/PARAFAC) tensor decomposition framework combined with PARAFAC2, CCC-RISE preserves the four-dimensional structure of multi-condition cell-cell communication data—conditions × sender cells × receiver cells × ligand-receptor pairs—instead of flattening it into conventional matrices.

CCC-RISE decomposes variation into distinct, interpretable patterns associated with experimental conditions, sender cells, receiver cells, and specific ligand-receptor interactions, providing a more nuanced and biologically meaningful analysis of intercellular communication dynamics. This approach enables the identification of coordinated signaling programs that change across conditions, offering insights into how cellular communication networks respond to different experimental perturbations.

CCC-RISE does not require prior cell-type labels or clustering, reducing bias and enabling discovery of novel communication patterns, while maintaining high resolution to identify condition-specific signaling subpopulations missed by pseudobulk or clustering-based approaches. Each resulting component is directly linked to specific conditions, sender cells, receiver cells, and ligand-receptor pairs, making the results biologically tractable.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   tutorial
   api
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Updated December 2025
