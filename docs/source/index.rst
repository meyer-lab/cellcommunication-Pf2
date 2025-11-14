.. cellcommunication-Pf2 documentation master file

CellCommunication-Pf2 Documentation
===================================

**CellCommunication-Pf2** is a computational biology tool for analyzing cell-cell 
communication using tensor decomposition methods. It combines RISE decomposition 
with CANDECOMP/PARAFAC (CP) tensor decomposition to identify communication patterns 
from single-cell RNA-seq data.

Key Features
------------

* **CCC-RISE Algorithm**: Combines RISE and CP decomposition for cell communication analysis
* **Ligand-Receptor Analysis**: Built-in database of protein-protein interactions
* **COVID-19 Dataset**: Pre-configured access to published BALF COVID-19 data
* **Stability Analysis**: Tools for evaluating decomposition robustness (FMS, R²X)
* **Pseudobulk Support**: Analysis at both single-cell and aggregated levels

Quick Links
-----------

* :doc:`quickstart` - Run your first analysis
* :doc:`api/index` - Complete API reference

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

