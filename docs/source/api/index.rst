.. _api:

API Reference
=============

Complete API documentation for all modules and functions.

Core Modules
------------

.. autosummary::
   :toctree: generated
   :recursive:

   cellcommunicationpf2.ccc_rise
   cellcommunicationpf2.tensor
   cellcommunicationpf2.import_data
   cellcommunicationpf2.utils
   cellcommunicationpf2.logreg
   cellcommunicationpf2.ccc

Main Analysis Functions
-----------------------

CCC-RISE Workflow
^^^^^^^^^^^^^^^^^

.. currentmodule:: cellcommunicationpf2

.. autosummary::
   :toctree: generated

   ccc_rise.ccc_rise
   tensor.run_ccc_rise_workflow
   ccc_rise.calc_communication_score
   ccc_rise.standardize_cc_pf2

Data Import
^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   import_data.import_balf_covid
   import_data.import_ligand_receptor_pairs
   import_data.import_alad
   import_data.add_cond_idxs
   import_data.prepare_dataset

Tensor Analysis
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   tensor.calculate_interaction_tensor
   tensor.run_fms_r2x_analysis
   tensor.run_fms_r2x_data_percentage_analysis
   tensor.calculate_fms_cpd
   tensor.calculate_fms_rise
   tensor.calculate_r2x

Utility Functions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   utils.resample
   utils.pseudobulk_X
   utils.correct_conditions
   utils.expression_product_matrix

Logistic Regression
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   logreg.logistic_regression
   logreg.rise_ranks_logreg
   logreg.cpd_ranks_logreg
   logreg.ccc_rise_logreg_weights

Detailed Module Documentation
------------------------------

.. toctree::
   :maxdepth: 2

   modules/ccc_rise
   modules/tensor
   modules/import_data
   modules/utils
   modules/logreg
   modules/ccc
