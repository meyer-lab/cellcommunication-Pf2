"""
Figure 1: XX
"""

from .common import getSetup, subplotLabel
from ..import_data import import_balf_covid, import_ligand_receptor_pairs
from scipy.sparse import issparse
import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp



def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)
    X = import_balf_covid()
    df_lrp = import_ligand_receptor_pairs()
    df_lrp = df_lrp[["ligand", "receptor"]]
    
    # Filter df_lrp to only include genes present in the data
    valid_mask = (df_lrp['ligand'].isin(X.var_names)) & (df_lrp['receptor'].isin(X.var_names))
    print(valid_mask)
    df_lrp = df_lrp[valid_mask].copy().reset_index(drop=True)

    
    # Get genes to keep
    genes_to_keep = list(set(df_lrp['ligand']) | set(df_lrp['receptor']))
    
    X = X[::200]
    X = X[:, genes_to_keep]
    # X = X[:, :20]
    print(X)
  
            
            

    

    
    XX = calculate_communication_scores_simple(X, df_lrp.iloc[:20, :])
    print(XX.obs_names)

    
    

    return f

import scipy.sparse as sp
import numpy as np
import pandas as pd
import anndata

import scipy.sparse as sp
import numpy as np
import pandas as pd
import anndata

def calculate_communication_scores_simple(adata, df_lrp, min_expr=0.0, same_cell=False):
    """
    Calculate cell-cell communication scores for specific L-R pairs.
    
    Parameters:
    - adata: AnnData object with filtered genes
    - df_lrp: DataFrame with filtered ligand-receptor pairs
    - min_expr: Minimum expression threshold (default 0.0)
    - same_cell: Whether to include same-cell interactions (default False)
    """
    results = []
    print(f"Processing {len(df_lrp)} L-R pairs")
    
    for sample in np.unique(adata.obs["sample_new"]):
        print(f"\nProcessing sample: {sample}")
        sample_data = adata[adata.obs["sample_new"] == sample]
        X_sparse = sample_data.X if sp.issparse(sample_data.X) else sp.csr_matrix(sample_data.X)
        
        # Generate cell pairs
        cell_pairs = [(s, r) for s in range(sample_data.shape[0]) 
                      for r in range(sample_data.shape[0]) 
                      if s != r or same_cell]
        print(f"Processing {len(cell_pairs)} cell pairs")
        
        # Calculate scores
        scores = sp.lil_matrix((len(cell_pairs), len(df_lrp)))
        for idx, (_, row) in enumerate(df_lrp.iterrows()):
            lig_idx = sample_data.var_names.get_loc(row['ligand'])
            rec_idx = sample_data.var_names.get_loc(row['receptor'])
            
            # Get expression vectors
            lig_vec = X_sparse[:, lig_idx].toarray().flatten()
            rec_vec = X_sparse[:, rec_idx].toarray().flatten()
            
            # Calculate scores for all cell pairs at once using broadcasting
            sender_expr = lig_vec[np.array([s for s, _ in cell_pairs])]
            receiver_expr = rec_vec[np.array([r for _, r in cell_pairs])]
            pair_scores = sender_expr * receiver_expr
            
            # Store scores
            scores[:, idx] = pair_scores
        
        # Store results
        for pair_idx, (s, r) in enumerate(cell_pairs):
            pair_scores = scores[pair_idx, :].toarray().flatten()
            results.append({
                'sample': sample,
                'sender': sample_data.obs.index[s],
                'receiver': sample_data.obs.index[r],
                'interaction_id': f"{sample_data.obs.index[s]}_{sample_data.obs.index[r]}_{sample}",
                'sender_type': sample_data.obs['celltype'].iloc[s],
                'receiver_type': sample_data.obs['celltype'].iloc[r],
                **{f'{row.ligand}_{row.receptor}': score 
                   for score, (_, row) in zip(pair_scores, df_lrp.iterrows())}
            })
    
    # Create AnnData object
    results_df = pd.DataFrame(results)
    score_columns = [f'{row.ligand}_{row.receptor}' for _, row in df_lrp.iterrows()]
    
    results_adata = anndata.AnnData(
        X=sp.csr_matrix(results_df[score_columns].values),
        obs=results_df.drop(columns=score_columns).set_index('interaction_id'),
        var=pd.DataFrame(index=score_columns)
    )
    
    return results_adata



# def prepare_dataset(
#     X: anndata.AnnData, condition_name: str, geneThreshold: float
# ) -> anndata.AnnData:
#     assert isinstance(X.X, csc_matrix | csr_matrix)
#     assert np.amin(X.X.data) >= 0.0

#     # Filter out genes with too few reads
#     readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
#     X = X[:, readmean > geneThreshold]

#     # Copy so that the subsetting is preserved
#     X._init_as_actual(X.copy())

#     # Normalize read depth
#     normalize_total(X)

#     # Scale genes by sum
#     readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
#     readsum = X.shape[0] * readmean
#     inplace_column_scale(X.X, 1.0 / readsum)  # type: ignore

#     # Transform values
#     X.X.data = np.log10((1000.0 * X.X.data) + 1.0)  # type: ignore

#     # Get the indices for subsetting the data
#     _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
#     X.obs["condition_unique_idxs"] = sgIndex

#     # Pre-calculate gene means
#     means, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
#     X.var["means"] = means

#     return X


# def anndata_to_3d_list(adata: 'anndata.AnnData', 
#                        condition_key: str = 'condition_unique_idxs',
#                        use_cupy: bool = False) -> list:
#     """
#     Converts an AnnData object to a 3D list structure based on a condition key.
    
#     Parameters
#     ----------
#     adata : anndata.AnnData
#         The AnnData object to convert.
#     condition_key : str, default='condition_unique_idxs'
#         The key in adata.obs to use for grouping the data.
#     use_cupy : bool, default=False
#         Whether to convert the arrays to CuPy arrays (for GPU processing).
        
#     Returns
#     -------
#     list
#         A 3D list structure where the first dimension represents conditions,
#         the second dimension represents observations within each condition,
#         and the third dimension represents features.
#     """
#     import numpy as np
#     import scipy.sparse as sps
    
#     # Ensure the condition key exists in adata.obs
#     if condition_key not in adata.obs.columns:
#         raise ValueError(f"The key '{condition_key}' is not found in adata.obs.")
    
#     # Extract the condition indices
#     condition_indices = adata.obs[condition_key].to_numpy(dtype=int)
#     n_conditions = np.max(condition_indices) + 1
    
#     # Initialize the 3D list
#     data_3d = []
    
#     # Process each condition
#     for i in range(n_conditions):
#         # Get indices for the current condition
#         mask = condition_indices == i
        
#         # Extract the corresponding data
#         if isinstance(adata.X, np.ndarray):
#             condition_data = adata.X[mask]
#         else:  # Assuming it's a sparse matrix
#             condition_data = adata.X[mask]
            
#             # Convert sparse matrix to dense if needed
#             # Uncomment the next line if you want to convert to dense
#             # condition_data = condition_data.toarray()
        
#         # Convert to CuPy if requested
#         if use_cupy:
#             import cupy as cp
#             import cupyx.scipy.sparse as cupy_sparse
            
#             if isinstance(condition_data, np.ndarray):
#                 condition_data = cp.array(condition_data)
#             else:  # Assuming it's a sparse matrix
#                 condition_data = cupy_sparse.csr_matrix(condition_data)
        
#         # Add to the 3D list
#         data_3d.append(condition_data)
    
#     return data_3d


# def anndata_to_communication_tensors(
#     adata: 'anndata.AnnData',
#     sender_key: str = 'sender_cell_type',
#     receiver_key: str = 'receiver_cell_type',
#     lr_pair_key: str = 'lr_pair',
#     condition_key: str = 'condition',
#     use_cupy: bool = False
# ) -> list:
#     """
#     Converts an AnnData object to a list of 4D tensors for cell-cell communication analysis.
    
#     Parameters
#     ----------
#     adata : anndata.AnnData
#         The AnnData object containing cell-cell communication data.
#     sender_key : str, default='sender_cell_type'
#         The key in adata.obs that identifies sender cell types.
#     receiver_key : str, default='receiver_cell_type'
#         The key in adata.obs that identifies receiver cell types.
#     lr_pair_key : str, default='lr_pair'
#         The key in adata.var that identifies ligand-receptor pairs.
#     condition_key : str, default='condition'
#         The key in adata.obs that identifies different experimental conditions.
#     use_cupy : bool, default=False
#         Whether to convert tensors to CuPy arrays (for GPU processing).
        
#     Returns
#     -------
#     list
#         A list of 4D tensors where each tensor has shape:
#         (n_sender_types, n_receiver_types, n_lr_pairs, n_samples_per_condition)
#         representing communication scores for each sender-receiver-lr_pair combination 
#         under each condition.
#     """
#     import numpy as np
#     import scipy.sparse as sps
#     from collections import defaultdict
    
#     # Ensure required keys exist
#     for key, container in [
#         (sender_key, 'obs'),
#         (receiver_key, 'obs'),
#         (condition_key, 'obs'),
#         (lr_pair_key, 'var')
#     ]:
#         if container == 'obs' and key not in adata.obs.columns:
#             raise ValueError(f"The key '{key}' is not found in adata.obs.")
#         elif container == 'var' and key not in adata.var.columns:
#             raise ValueError(f"The key '{key}' is not found in adata.var.")
    
#     # Extract unique values for each dimension
#     sender_types = adata.obs[sender_key].unique()
#     receiver_types = adata.obs[receiver_key].unique()
#     lr_pairs = adata.var[lr_pair_key].unique()
#     conditions = adata.obs[condition_key].unique()
    
#     # Create mappings for faster indexing
#     sender_map = {val: idx for idx, val in enumerate(sender_types)}
#     receiver_map = {val: idx for idx, val in enumerate(receiver_types)}
#     lr_map = {val: idx for idx, val in enumerate(lr_pairs)}
#     condition_map = {val: idx for idx, val in enumerate(conditions)}
    
#     # Initialize the tensor list
#     tensor_list = []
    
#     # Process each condition separately
#     for condition in conditions:
#         # Create a 4D tensor for this condition
#         # Shape: (n_sender_types, n_receiver_types, n_lr_pairs, n_samples_per_condition)
#         # First, determine number of samples in this condition
#         condition_mask = adata.obs[condition_key] == condition
#         n_samples = np.sum(condition_mask)
        
#         # Initialize the tensor
#         tensor_shape = (len(sender_types), len(receiver_types), len(lr_pairs), n_samples)
#         condition_tensor = np.zeros(tensor_shape)
        
#         # Group observations by sender and receiver types
#         for sender_type in sender_types:
#             sender_idx = sender_map[sender_type]
#             for receiver_type in receiver_types:
#                 receiver_idx = receiver_map[receiver_type]
                
#                 # Find observations for this sender-receiver pair in this condition
#                 pair_mask = (adata.obs[sender_key] == sender_type) & \
#                             (adata.obs[receiver_key] == receiver_type) & \
#                             condition_mask
                
#                 if np.sum(pair_mask) == 0:
#                     continue  # No data for this combination
                
#                 # Extract data for this sender-receiver pair
#                 if isinstance(adata.X, np.ndarray):
#                     pair_data = adata.X[pair_mask]
#                 else:  # Sparse matrix
#                     pair_data = adata.X[pair_mask].toarray()
                
#                 # Map each LR pair column to the correct index in the tensor
#                 for lr_pair in lr_pairs:
#                     lr_idx = lr_map[lr_pair]
                    
#                     # Find columns corresponding to this LR pair
#                     lr_col_mask = adata.var[lr_pair_key] == lr_pair
                    
#                     if np.sum(lr_col_mask) == 0:
#                         continue  # No data for this LR pair
                    
#                     # Extract and store the values
#                     lr_data = pair_data[:, lr_col_mask]
#                     condition_tensor[sender_idx, receiver_idx, lr_idx, :lr_data.shape[0]] = lr_data.flatten()[:n_samples]
        
#         # Convert to CuPy if requested
#         if use_cupy:
#             import cupy as cp
#             condition_tensor = cp.array(condition_tensor)
        
#         # Add to the list
#         tensor_list.append(condition_tensor)
    
#     return tensor_list