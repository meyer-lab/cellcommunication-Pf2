CCC Module
==========

.. automodule:: cellcommunicationpf2.ccc
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   This module contains methods adapted from Tensor Cell2Cell for computing
   cell-cell communication scores.

Communication Score Functions
-----------------------------

.. autofunction:: cellcommunicationpf2.ccc.compute_ccc_matrix

.. autofunction:: cellcommunicationpf2.ccc.aggregate_ccc_matrices

.. autofunction:: cellcommunicationpf2.ccc.aggregate_ccc_tensor

Tensor Building Functions
--------------------------

.. autofunction:: cellcommunicationpf2.ccc.build_context_ccc_tensor

.. autofunction:: cellcommunicationpf2.ccc.generate_ccc_tensor

Filtering Functions
-------------------

.. autofunction:: cellcommunicationpf2.ccc.filter_ppi_by_proteins

.. autofunction:: cellcommunicationpf2.ccc.filter_complex_ppi_by_proteins

Helper Functions
----------------

.. autofunction:: cellcommunicationpf2.ccc.get_genes_from_complexes

.. autofunction:: cellcommunicationpf2.ccc.get_element_abundances

.. autofunction:: cellcommunicationpf2.ccc.get_elements_over_fraction
