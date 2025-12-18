API Reference
=============

Core Functions
--------------

Preprocessing
~~~~~~~~~~~~~

.. py:function:: prepare_dataset(X: anndata.AnnData, condition_name: str, geneThreshold: float, normalize: bool = False)
   :module: cellcommunicationpf2.import_data

   Preprocess single-cell RNA-seq data for CCC-RISE analysis.
   
   This function performs essential preprocessing steps including cell and gene filtering, 
   normalization, log transformation, and creation of condition indices required for 
   CCC-RISE decomposition.
   
   :param X: AnnData object containing raw count data in sparse matrix format.
   :type X: anndata.AnnData
   :param condition_name: Name of the column in X.obs that specifies experimental conditions for each cell.
   :type condition_name: str
   :param geneThreshold: Minimum mean expression threshold for gene filtering. Genes with mean expression below this value are removed.
   :type geneThreshold: float
   :param normalize: If True, performs normalization and log transformation. If False, keeps raw counts.
   :type normalize: bool, optional
   :returns: Preprocessed AnnData object with filtered cells/genes, added condition_unique_idxs column, and gene means in X.var["means"].
   :rtype: anndata.AnnData

.. py:function:: import_ligand_receptor_pairs(filename: str = "/opt/andrew/ccc/Human-2020-Jin-LR-pairs.csv.zst", update_interaction_names: bool = True)
   :module: cellcommunicationpf2.import_data

   Import ligand-receptor pairs from a compressed CSV file with caching.
   
   Loads a curated database of ligand-receptor pairs from CellChat. The function uses 
   LRU caching to avoid repeated file reads. For protein complexes, subunits are separated 
   by & (e.g., CD74&CD44).
   
   :param filename: Path to the compressed CSV file containing ligand-receptor pairs.
   :type filename: str, optional
   :param update_interaction_names: If True, updates interaction names to use standardized formatting (uppercase, & for complexes).
   :type update_interaction_names: bool, optional
   :returns: DataFrame with ligand and receptor columns containing gene names.
   :rtype: pd.DataFrame

.. py:function:: add_cond_idxs(X: anndata.AnnData, condition_key: str)
   :module: cellcommunicationpf2.import_data

   Add unique 0-indexed condition identifiers to an AnnData object.
   
   Creates a new column condition_unique_idxs in X.obs that maps condition labels 
   to 0-based integer indices. This is required for proper data partitioning in the 
   PARAFAC2 decomposition.
   
   :param X: AnnData object to add condition indices to.
   :type X: anndata.AnnData
   :param condition_key: Column name in X.obs containing condition identifiers.
   :type condition_key: str
   :returns: AnnData object with added condition_unique_idxs column in X.obs.
   :rtype: anndata.AnnData


Factorization
~~~~~~~~~~~~~

.. py:function:: run_ccc_rise_workflow(adata: anndata.AnnData, rise_rank: int, lr_pairs: pd.DataFrame, cp_rank: int = None, condition_column: str = "sample", n_iter_max: int = 100, tol: float = 1e-3, random_state: int = None, complex_sep: str = None, doEmbedding: bool = True, svd_init: str = "svd")
   :module: cellcommunicationpf2.tensor

   Execute the complete CCC-RISE workflow including RISE decomposition, CPD factorization, and result storage.
   
   This is the main function for performing CCC-RISE analysis. It executes both the RISE 
   (PARAFAC2) decomposition of expression data and the CPD decomposition of the resulting 
   interaction tensor. The function decomposes cell-cell communication into four interpretable 
   factor matrices representing conditions, sender cells, receiver cells, and ligand-receptor pairs.
   
   :param adata: AnnData object with preprocessed scRNA-seq data. Must have condition_unique_idxs in adata.obs.
   :type adata: anndata.AnnData
   :param rise_rank: Number of PARAFAC2 components to extract from expression data. Typically chosen based on FMS and R²X analysis.
   :type rise_rank: int
   :param lr_pairs: DataFrame of ligand-receptor pairs with 'ligand' and 'receptor' columns.
   :type lr_pairs: pd.DataFrame
   :param cp_rank: Number of CPD components for factorizing the interaction tensor. If None, defaults to rise_rank.
   :type cp_rank: int, optional
   :param condition_column: Column name in adata.obs containing condition identifiers.
   :type condition_column: str, optional
   :param n_iter_max: Maximum iterations for decomposition.
   :type n_iter_max: int, optional
   :param tol: Convergence tolerance for optimization.
   :type tol: float, optional
   :param random_state: Random seed for reproducibility of decomposition.
   :type random_state: int, optional
   :param complex_sep: Separator for protein complexes in L-R pairs (typically "&").
   :type complex_sep: str, optional
   :param doEmbedding: If True, automatically computes PaCMAP embeddings for visualization and stores in adata.obsm["PaCMAP"].
   :type doEmbedding: bool, optional
   :param svd_init: Initialization method for CPD ('svd' or 'random').
   :type svd_init: str, optional
   :returns: Tuple containing the updated AnnData object with stored results and the R²X value (variance explained).
   :rtype: tuple[anndata.AnnData, float]

.. py:function:: calculate_interaction_tensor(X_filtered: anndata.AnnData, lr_pairs: pd.DataFrame, rise_rank: int)
   :module: cellcommunicationpf2.tensor

   Calculate the interaction tensor from AnnData object using PARAFAC2 and communication scores.
   
   This function performs RISE decomposition and computes cell-cell communication scores for 
   all ligand-receptor pairs across sender-receiver latent cell state pairs. The resulting 
   tensor has dimensions (rise_rank × rise_rank × n_lr_pairs × n_conditions).
   
   :param X_filtered: AnnData object containing preprocessed expression data.
   :type X_filtered: anndata.AnnData
   :param lr_pairs: DataFrame of ligand-receptor pairs with 'ligand' and 'receptor' columns.
   :type lr_pairs: pd.DataFrame
   :param rise_rank: Number of PARAFAC2 components to extract before computing communication scores.
   :type rise_rank: int
   :returns: Interaction tensor with dimensions (rise_rank × rise_rank × n_lr_pairs × n_conditions).
   :rtype: np.ndarray

.. py:function:: run_fms_r2x_analysis(interaction_tensor: np.ndarray, rank_list: list[int] = None, runs: int = 1, svd_init: str = "svd")
   :module: cellcommunicationpf2.tensor

   Run Factor Match Score (FMS) and R²X analysis across different CPD ranks to assess stability and fit.
   
   This function helps determine the optimal CPD rank by evaluating model stability (FMS) 
   through bootstrap resampling and variance explained (R²X). FMS values above 0.6 indicate 
   reliable, reproducible components. Use this before finalizing your CPD rank choice.
   
   :param interaction_tensor: Pre-computed interaction tensor from calculate_interaction_tensor().
   :type interaction_tensor: np.ndarray
   :param rank_list: List of CPD ranks to test (e.g., [1, 3, 5, 7, 9]). If None, defaults to [1, 3].
   :type rank_list: list of int, optional
   :param runs: Number of bootstrap runs for stability assessment.
   :type runs: int, optional
   :param svd_init: Initialization method ('svd' or 'random').
   :type svd_init: str, optional
   :returns: DataFrame with columns ['Run', 'Component', 'FMS', 'R2X'] for each rank and run.
   :rtype: pd.DataFrame


Visualization Functions
-----------------------

Rank Selection
~~~~~~~~~~~~~~

.. py:function:: plot_fms_r2x_diff_ranks(X: anndata.AnnData, condition_name: str, ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, ranksList: list[int], runs: int)
   :module: cellcommunicationpf2.figures.commonFuncs.plotGeneral

   Plot Factor Match Score (FMS) and R²X across different RISE ranks for rank selection.
   
   This function evaluates multiple RISE ranks by computing FMS (stability) and R²X 
   (variance explained) metrics. It performs bootstrap resampling to assess component 
   reproducibility. Use this to determine the optimal RISE rank before running the full 
   CCC-RISE workflow.
   
   :param X: AnnData object with preprocessed data. Must have condition_unique_idxs in X.obs.
   :type X: anndata.AnnData
   :param condition_name: Column name in X.obs containing condition identifiers for bootstrap resampling.
   :type condition_name: str
   :param ax1: Matplotlib axes object for plotting FMS values.
   :type ax1: matplotlib.axes.Axes
   :param ax2: Matplotlib axes object for plotting R²X values.
   :type ax2: matplotlib.axes.Axes
   :param ranksList: List of RISE ranks to evaluate (e.g., [5, 10, 15, 20, 25, 30, 35, 40]).
   :type ranksList: list of int
   :param runs: Number of bootstrap runs for stability assessment per rank.
   :type runs: int


Factor Plotting
~~~~~~~~~~~~~~~

.. py:function:: plot_condition_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, cond: str = "Condition", cond_group_labels: pd.Series = None, color_key: list = None, group_cond: bool = False, normalize: bool = False)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor A (condition factors) as a heatmap showing how conditions contribute to components.
   
   This visualization shows how each experimental condition (rows) contributes to each CCC-RISE 
   component (columns). High values indicate strong association between a condition and a 
   component's communication pattern.

   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["A"] and data.obs[cond].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cond: Name of column in data.obs containing condition labels.
   :type cond: str, optional
   :param cond_group_labels: Series mapping conditions to group labels for colored row annotations.
   :type cond_group_labels: pd.Series, optional
   :param color_key: Custom colors for condition group labels.
   :type color_key: list, optional
   :param group_cond: If True and cond_group_labels provided, sorts conditions by group.
   :type group_cond: bool, optional
   :param normalize: If True, normalizes each component to [-1, 1] range.
   :type normalize: bool, optional

.. py:function:: plot_eigenstate_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, factor_type: str)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor B (sender eigenstates) or Factor C (receiver eigenstates) as a heatmap.
   
   Eigenstate factors represent the underlying cell state patterns across components in the 
   latent RISE space. Each row represents a latent dimension from RISE and each column 
   represents a CCC-RISE component.
   
   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["B"] or data.uns["C"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param factor_type: Either "B" for sender eigenstates or "C" for receiver eigenstates.
   :type factor_type: str

.. py:function:: plot_lr_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, trim: bool = True, weight: float = 0.08)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor D (ligand-receptor pairs) as a heatmap showing which L-R pairs drive each component.
   
   This visualization reveals coordinated signaling programs by showing which ligand-receptor 
   pairs (rows) are highly weighted in each component (columns). Only L-R pairs with maximum 
   absolute weight above the threshold are displayed.
   
   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["D"] and data.uns["lr_pairs"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param trim: If True, filters L-R pairs based on the weight parameter.
   :type trim: bool, optional
   :param weight: Minimum absolute weight threshold for including L-R pairs.
   :type weight: float, optional

.. py:function:: plot_lr_factors_partial(X: anndata.AnnData, cmp: int, ax: matplotlib.axes.Axes, geneAmount: int = 5, top: bool = True)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot the top or bottom weighted ligand-receptor pairs for a specific component as a bar plot.
   
   This visualization identifies the most positively or negatively weighted L-R pairs for a 
   single component, revealing which specific interactions are most associated with that 
   communication pattern.
   
   :param X: AnnData object with stored CCC-RISE results. Must contain X.uns["D"] and X.uns["lr_pairs"].
   :type X: anndata.AnnData
   :param cmp: Component number to visualize (1-indexed).
   :type cmp: int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param geneAmount: Number of L-R pairs to display.
   :type geneAmount: int, optional
   :param top: If True, shows highest-weighted pairs; if False, shows lowest-weighted pairs.
   :type top: bool, optional


PaCMAP Visualization
~~~~~~~~~~~~~~~~~~~~

.. py:function:: plot_labels_pacmap(X: anndata.AnnData, labelType: str, ax: matplotlib.axes.Axes, condition: list = None, cmap: str = "tab20", color_key: list = None)
   :module: cellcommunicationpf2.figures.commonFuncs.plotPaCMAP

   Plot PaCMAP embedding colored by categorical labels (cell type or condition).
   
   This visualization shows the overall structure of the cell embedding in the latent 
   communication space, revealing how cells cluster by cell type or experimental condition 
   based on their communication patterns.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["PaCMAP"] and X.obs[labelType].
   :type X: anndata.AnnData
   :param labelType: Name of column in X.obs containing categorical labels to color by.
   :type labelType: str
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param condition: If provided, only highlights cells from these specific conditions.
   :type condition: list of str, optional
   :param cmap: Matplotlib colormap name for coloring categories.
   :type cmap: str, optional
   :param color_key: Custom list of colors for categories.
   :type color_key: list, optional

.. py:function:: plot_wc_pacmap(X: anndata.AnnData, cmp: int, ax: matplotlib.axes.Axes, cbarMax: float = 1.0, factor_matrix: str = None)
   :module: cellcommunicationpf2.figures.commonFuncs.plotPaCMAP

   Plot PaCMAP embedding colored by weighted projections for a specific component.
   
   This visualization shows which cells contribute most strongly to a specific component by 
   coloring them according to their sender (Factor B) or receiver (Factor C) weights. Higher 
   values indicate stronger association with the component's communication pattern.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["PaCMAP"] and X.obsm["sc_B"] or X.obsm["rc_C"].
   :type X: anndata.AnnData
   :param cmp: Component number to visualize (1-indexed).
   :type cmp: int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cbarMax: Maximum value for the color scale.
   :type cbarMax: float, optional
   :param factor_matrix: Either "B" for sender weights or "C" for receiver weights.
   :type factor_matrix: str


Analysis Functions
~~~~~~~~~~~~~~~~~~

.. py:function:: expression_product_matrix(X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str)
   :module: cellcommunicationpf2.utils

   Calculate the expression product matrix for a specific ligand-receptor pair between cell populations.
   
   For each cell in X1 (senders) and each cell in X2 (receivers), this computes the product 
   of ligand expression (in sender) and receptor expression (in receiver). This represents 
   the potential for that specific L-R interaction between each sender-receiver cell pair.
   
   :param X1: AnnData object containing sender cells.
   :type X1: anndata.AnnData
   :param X2: AnnData object containing receiver cells.
   :type X2: anndata.AnnData
   :param ligand: Ligand gene name (must be present in X1.var_names).
   :type ligand: str
   :param receptor: Receptor gene name (must be present in X2.var_names).
   :type receptor: str
   :returns: DataFrame with sender cells as rows and receiver cells as columns, values are expression products.
   :rtype: pd.DataFrame

.. py:function:: average_product_matrix_ccc(df: pd.DataFrame)
   :module: cellcommunicationpf2.utils

   Reduce an expression product matrix to a 10×10 matrix by binning rows and columns and averaging.
   
   This function groups rows and columns into 10 bins each and computes the mean within 
   each bin, creating a summarized heatmap suitable for visualization of large cell-cell 
   interaction matrices.
   
   :param df: Expression product matrix from expression_product_matrix().
   :type df: pd.DataFrame
   :returns: 10×10 DataFrame with averaged expression products.
   :rtype: pd.DataFrame
