.. CCC-RISE documentation master file

CCC-RISE Documentation
======================

CCC-RISE is a computational tool for analyzing cell-cell communication across 
experimental conditions or contexts at single-cell resolution. CCC-RISE is based 
on the unsupervised tensor decomposition PARAFAC2, which has been tailored to 
analyze scRNA-seq data through Reduction and Insight in Single-cell Exploration 
(RISE). CCC-RISE deconvolves signaling patterns into modules across sender cells, 
receiver cells, ligand-receptor pairs, and experimental conditions or samples.

Quick Links
-----------

* :doc:`quickstart` - Run your first analysis
* :doc:`api/index` - Complete API reference

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   preprocessing
   rank_selection
   scoring_methods
   pacmap_guidance

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
