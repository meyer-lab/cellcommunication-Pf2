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
    
    df_lrp = import_ligand_receptor_pairs()
    df_lrp = df_lrp[["ligand", "receptor"]]
    lr_genes = set(df_lrp['ligand'].unique()) | set(df_lrp['receptor'].unique())

    X = import_balf_covid()
    print(X)
    if sp.issparse(X.X):
        X = X[:500, ::1000]
        total = X.X.shape[0] * X.X.shape[1]
        nonzeros = np.count_nonzero(X.X.toarray())
        zeros = total - nonzeros
        
        # Print detailed information
        print(f"\nMatrix analysis:")
        print(f"Total entries: {total:,}")
        print(f"Non-zero entries: {nonzeros:,}")
        print(f"Zero entries: {zeros:,}")
        print(f"Sparsity: {zeros/total:.4f} ({zeros/total*100:.2f}%)")
            
            
    # total_entries = X.X.size
    # non_zero_entries = X.X.nnz  # number of non-zero elements
    # zero_entries = total_entries - non_zero_entries
    # zero_percentage = (zero_entries / total_entries) * 100
    
    # print("\nZero values analysis:")
    # print(f"Matrix shape: {X.X.shape}")
    # print(f"Total entries: {total_entries:,}")
    # print(f"Non-zero entries: {total_entries - zero_entries:,}")
    # print(f"Zero entries: {zero_entries:,}")
    # print(f"Percentage zeros: {zero_percentage:.2f}%")
    
    # genes_to_keep = [gene for gene in X.var_names if gene in lr_genes]
    # X = X[:, genes_to_keep]
    # print(X)
    # print(X.obs["sample"].unique())
    # print(X.obs["sample_new"].unique())
    # # X.X = X.X.toarray() if issparse(X.X) else X.X
    
    

    
    # XX = calculate_communication_scores(X, df_lrp)
    # print(XX)
    # total_entries = XX.X.size
    # zero_entries = (XX.X == 0).sum()
    # zero_percentage = (zero_entries / total_entries) * 100
    
    # print("\nZero values analysis:")
    # print(f"Matrix shape: {XX.X.shape}")
    # print(f"Total entries: {total_entries:,}")
    # print(f"Non-zero entries: {total_entries - zero_entries:,}")
    # print(f"Zero entries: {zero_entries:,}")
    # print(f"Percentage zeros: {zero_percentage:.2f}%")
    
    # # print(XX.obs["sender_type"].unique())
    # # print(len(XX.obs["sender_type"].unique()))
    # # print(len(XX.obs["receiver_type"].unique()))
    
    # for i in X.obs["sample"].unique():
    #     X_sample = X[X.obs["sample"] == i]
    #     print(X_sample.shape)

    
    

    return f

def calculate_communication_scores(adata, df_lrp):
    """Calculate cell-cell communication scores based on ligand-receptor pairs."""
    
    results = []
    lr_pairs = [f"{row['ligand']}-{row['receptor']}" for _, row in df_lrp.iterrows()]
    print(f"\nTotal L-R pairs in database: {len(df_lrp)}")
    print("Available genes in data:", len(adata.var_names))
    print(adata.var_names)
    
    
    # Process each sample
    for i in adata.obs["sample"].unique():
        print(f"\nProcessing sample: {i}")
        X_sample = adata[adata.obs["sample"] == i]
        
        valid_pairs = []
        for _, row in df_lrp.iterrows():
            ligand, receptor = row['ligand'], row['receptor']
            if ligand in adata.var_names and receptor in adata.var_names:
                valid_pairs.append(f"{ligand}-{receptor}")
        
        
        # Convert sparse matrix to dense for this sample
        if issparse(X_sample.X):
            X_dense = X_sample.X.toarray()
        else:
            X_dense = X_sample.X
            
        # Get metadata for cell pairs
        cell_indices = np.arange(X_sample.shape[0])
        sender_cells, receiver_cells = np.meshgrid(cell_indices, cell_indices)
        mask = sender_cells != receiver_cells
        
        # Store metadata for valid cell pairs
        pairs_metadata = [{
            'sender_cell': X_sample.obs.index[s],
            'receiver_cell': X_sample.obs.index[r],
            'sender_type': X_sample.obs['celltype'].iloc[s],
            'receiver_type': X_sample.obs['celltype'].iloc[r],
            'sample': i,
            'disease': X_sample.obs['disease'].iloc[r],
            'group': X_sample.obs['group'].iloc[r],
            'hasnCoV': X_sample.obs['hasnCoV'].iloc[r],
            'cluster': X_sample.obs['cluster'].iloc[r],
            'condition': X_sample.obs['condition'].iloc[r],
            'interaction_id': f"{X_sample.obs.index[s]}_{X_sample.obs.index[r]}_{i}"
        } for s, r in zip(sender_cells[mask], receiver_cells[mask])]
        
        # print(pairs_metadata)
        # Calculate scores for all L-R pairs at once
        sample_scores = []
        # print(df_lrp)
        for _, row in df_lrp.iterrows():
            ligand, receptor = row['ligand'], row['receptor']
            # print(ligand, receptor)
            
            if ligand in adata.var_names and receptor in adata.var_names:
                lig_idx = X_sample.var_names.get_loc(ligand)
                rec_idx = X_sample.var_names.get_loc(receptor)
                
                # Get all ligand and receptor expressions
                lig_expr = X_dense[:, lig_idx]
                rec_expr = X_dense[:, rec_idx]
                
                # Calculate scores for all cell pairs at once using outer product
                pair_scores = np.outer(lig_expr, rec_expr)[mask]
                sample_scores.append(pair_scores)
        
        # Stack scores for all L-R pairs
        sample_scores = np.column_stack(sample_scores)
        
        # Add metadata and scores for this sample
        results.extend([{**meta, **{f'score_{lr}': score for lr, score in zip(lr_pairs, scores)}}
                       for meta, scores in zip(pairs_metadata, sample_scores)])
    
    # Create DataFrame and convert to AnnData
    results_df = pd.DataFrame(results).set_index('interaction_id')
    # Separate scores from metadata
    score_columns = [col for col in results_df.columns if col.startswith('score_')]
    scores_matrix = results_df[score_columns].values
    metadata_df = results_df.drop(columns=score_columns)
    
    # Create AnnData object
    results_adata = anndata.AnnData(
        X=scores_matrix,
        obs=metadata_df,
        var=pd.DataFrame(index=[col.replace('score_', '') for col in score_columns])
    )
    
    return results_adata




# def calculate_communication_scores(adata, df_lrp):
#     """Calculate cell-cell communication scores based on all ligand-receptor pairs."""
    
#     results = []
    
#     # Process each sample
#     for i in adata.obs["sample"].unique():
#         X_sample = adata[adata.obs["sample"] == i]
        
#         # Convert sparse matrix to dense for this sample
#         X_dense = X_sample.X.toarray() if issparse(X_sample.X) else X_sample.X
            
#         # Get metadata for cell pairs
#         cell_indices = np.arange(X_sample.shape[0])
#         sender_cells, receiver_cells = np.meshgrid(cell_indices, cell_indices)
#         mask = sender_cells != receiver_cells
        
#         # Initialize arrays to store sums for Jaccard calculation
#         numerator_sum = np.zeros(mask.sum())
#         denominator_sum = np.zeros(mask.sum())
        
#         # Process each L-R pair and accumulate scores
#         for _, row in df_lrp.iterrows():
#             ligand, receptor = row['ligand'], row['receptor']
            
#             if ligand in X_sample.var_names and receptor in X_sample.var_names:
#                 lig_idx = X_sample.var_names.get_loc(ligand)
#                 rec_idx = X_sample.var_names.get_loc(receptor)
                
#                 # Get expression values
#                 lig_expr = X_dense[:, lig_idx]
#                 rec_expr = X_dense[:, rec_idx]
                
#                 # Calculate components for Jaccard score
#                 numerator = np.outer(lig_expr, rec_expr)[mask]
#                 denominator = (np.outer(lig_expr, lig_expr) + 
#                              np.outer(rec_expr, rec_expr) - 
#                              numerator)[mask]
                
#                 # Accumulate sums
#                 numerator_sum += numerator
#                 denominator_sum += denominator
        
#         # Calculate final Jaccard scores
#         cci_scores = np.zeros_like(numerator_sum)
#         valid_mask = denominator_sum != 0
#         cci_scores[valid_mask] = numerator_sum[valid_mask] / denominator_sum[valid_mask]
#         cci_scores = np.nan_to_num(cci_scores, nan=0.0)
        
#         # Store metadata and scores for valid cell pairs
#         for idx, (s, r) in enumerate(zip(sender_cells[mask], receiver_cells[mask])):
#             results.append({
#                 'sender_cell': X_sample.obs.index[s],
#                 'receiver_cell': X_sample.obs.index[r],
#                 'sender_type': X_sample.obs['celltype'].iloc[s],
#                 'receiver_type': X_sample.obs['celltype'].iloc[r],
#                 'sample': i,
#                 'disease': X_sample.obs['disease'].iloc[r],
#                 'group': X_sample.obs['group'].iloc[r],
#                 'hasnCoV': X_sample.obs['hasnCoV'].iloc[r],
#                 'cluster': X_sample.obs['cluster'].iloc[r],
#                 'condition': X_sample.obs['condition'].iloc[r],
#                 'interaction_id': f"{X_sample.obs.index[s]}_{X_sample.obs.index[r]}_{i}",
#                 'cci_score': cci_scores[idx]
#             })
    
#     # Create DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Create AnnData object
#     results_adata = anndata.AnnData(
#         X=results_df['cci_score'].values.reshape(-1, 1),
#         obs=results_df.drop(columns=['cci_score']).set_index('interaction_id'),
#         var=pd.DataFrame(index=['cci_score'])
#     )
    
#     return results_adata