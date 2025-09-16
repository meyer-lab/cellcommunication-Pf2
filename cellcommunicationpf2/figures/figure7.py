# """
# Figure 7: RISE Projection Boxplots by Cell Type

# This figure runs RISE stacks the projections, and for each eigen-state,
# plots boxplots of the projection values for each cell type.
# """

# import numpy as np
# import pandas as pd

# from parafac2.parafac2 import parafac2_nd
# from ..import_data import import_balf_covid, add_cond_idxs
# from .common import getSetup, subplotLabel


# def makeFigure():
#     # Parameters
#     n_eigenstates = 30

#     # Load and prepare data
#     print("Importing BALF COVID-19 data...")
#     adata = import_balf_covid()
#     adata = add_cond_idxs(adata, "sample")
#     print("Loaded data with shape:", adata.shape)

#     # Run RISE (pf2_nd)
#     print("Running RISE...")
#     pf2_output, _ = parafac2_nd(
#         adata, rank=n_eigenstates, n_iter_max=100, tol=1e-6, random_state=42
#     )
#     _, _, projections = pf2_output
#     stacked = np.concatenate(
#         [proj.T for proj in projections], axis=1
#     )  # (n_eigenstates, total_cells)
#     print(f"Stacked projections shape: {stacked.shape}")

#     # Get cell type labels for all cells (in the same order as stacking)
#     celltype_labels = np.concatenate(
#         [
#             adata.obs.loc[adata.obs["condition_unique_idxs"] == i, "celltype"].values
#             for i in range(len(projections))
#         ]
#     )
#     unique_celltypes = pd.Categorical(celltype_labels).categories

#     # Use a fixed x-axis range for all plots to highlight differences
#     min_val, max_val = -0.5, 0.5

#     # Setup figure: 6 rows x 5 columns
#     nrows, ncols = 6, 5
#     ax, f = getSetup((3 * ncols, 2 * nrows), (nrows, ncols))
#     subplotLabel(ax)

#     # For each eigen-state, plot horizontal boxplots for all cell types
#     for r in range(n_eigenstates):
#         values = stacked[r, :]
#         data = [values[celltype_labels == ct] for ct in unique_celltypes]
#         ax[r].boxplot(data, vert=False, labels=unique_celltypes, showfliers=False)
#         # Remove subplot title, clarify axis label
#         if r % ncols == 0:
#             ax[r].set_ylabel("Cell Type")
#         else:
#             ax[r].set_yticklabels([])
#         ax[r].set_xlabel(f"Eigen-state {r+1} Projection Value")
#         ax[r].set_xlim(min_val, max_val)

#     # Remove the overall figure title for a cleaner look
#     # f.suptitle("RISE Projections by Cell Type and Rank", fontsize=16)
#     return f
