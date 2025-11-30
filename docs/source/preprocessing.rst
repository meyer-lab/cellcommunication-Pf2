Preprocessing
=============

This page documents the preprocessing steps the code expects and why they
matter. Small changes to filtering and normalization often have large effects
on downstream communication analysis.

What the code currently does
----------------------------

The implementation of `prepare_dataset` (in `cellcommunicationpf2.import_data`) performs
the following exact steps:

1. Assert that `X.X` is sparse. If `X.raw` is present and not None, the code uses `X.raw.X`; otherwise it uses `X.X`.

2. Convert the count matrix to `csr_array` and check there are no negative
   values.

3. Filter out genes and cells using two thresholds:

   - Keep cells with total counts > 10 (rows with sum > 10).
   - Keep genes whose mean expression across cells is > `geneThreshold`.

4. Copy the reduced AnnData to preserve subsetting and convert counts to
   floats if necessary.

5. If `normalize=True`, apply the following transformations in-place:

   - Scale `counts_per_cell` by dividing by the median counts per cell, then
     scale CSR matrix rows by `counts_per_cell`.
   - Scale genes by column sums.
   - Apply `log10(1000 * x + 1)` to the data values.

6. Compute `X.obs['condition_unique_idxs']` by relabeling the `condition_name`
   column into integer indices.

7. Store `X.var['means']` as the mean expression per gene.

Suggested checks before running CCC-RISE
----------------------------------------

1. Confirm that gene names in `adata.var_names` match the naming convention in
   your ligand–receptor table (case/underscore differences can break matching).
2. Run a small diagnostic: compute the number of genes kept by `geneThreshold`
   and inspect that useful marker genes are not filtered out.

Mapping genes to ligand-receptor pairs
--------------------------------------

The `import_ligand_receptor_pairs` helper reads a ligand–receptor table and
attempts to populate `ligand`, `receptor`, and `interaction_symbol` columns.
When `update_interaction_names=True` (the default), the code upper-cases the
ligand/receptor names from `interaction_name_2` or `interaction_symbol` columns
for matching. When the table encodes complexes, the `get_genes_from_complexes`
helper will split complex names on the configured separator (default '&').
Since gene matching is case-sensitive, ensure your `var_names` use the same
case (typically uppercase) or normalize them prior to running the pipeline.

Common pitfalls
---------------

- Mismatched gene naming (case, alias symbols): ensure `adata.var_names` and
  the LR table use the same naming.

If you hit any of these issues and are unsure how to proceed, mark the
examples below with your dataset specifics and open an issue or consult the
paper for rationale.
