"""
Figure 1: XX
"""

from .common import getSetup, subplotLabel
from ..import_data import import_balf_covid, import_ligand_receptor_pairs
from scipy.sparse import issparse
import numpy as np
import pandas as pd
import anndata


def makeFigure():
    ax, f = getSetup((12, 12), (3, 3))
    subplotLabel(ax)
    
    df_lrp = import_ligand_receptor_pairs()
    df_lrp = df_lrp[["ligand", "receptor"]]
    lr_genes = set(df_lrp['ligand'].unique()) | set(df_lrp['receptor'].unique())
    
    print(df_lrp)
    
    X = import_balf_covid()
    X = X[::1000, :2000]
    
    genes_to_keep = [gene for gene in X.var_names if gene in lr_genes]
    X = X[:, genes_to_keep]
    print(X)
    print(X.obs["sample"].unique())
    print(X.obs["sample_new"].unique())
    # X.X = X.X.toarray() if issparse(X.X) else X.X
    
    

    
    XX = calculate_communication_scores(X, df_lrp)
    print(XX)
    
    print(XX.obs["sender_type"].unique())
    print(len(XX.obs["sender_type"].unique()))
    print(len(XX.obs["receiver_type"].unique()))

    
    

    return f

def calculate_communication_scores(adata, df_lrp):
    """Calculate cell-cell communication scores based on ligand-receptor pairs."""
    
    results = []
    # Create a unique identifier for each ligand-receptor pair
    lr_pairs = [f"{row['ligand']}-{row['receptor']}" for _, row in df_lrp.iterrows()]
    
    for i in adata.obs["sample"].unique():
        X_sample = adata[adata.obs["sample"] == i]
        
        # Convert sparse matrix to dense for this sample
        if issparse(X_sample.X):
            X_dense = X_sample.X.toarray()
        else:
            X_dense = X_sample.X
        
        # Get all possible cell pairs
        cell_indices = np.arange(X_sample.shape[0])
        sender_cells, receiver_cells = np.meshgrid(cell_indices, cell_indices)
        mask = sender_cells != receiver_cells
        
        # Store metadata for valid cell pairs
        for s, r in zip(sender_cells[mask], receiver_cells[mask]):
            results.append({
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
            })
    
    # Create DataFrame for cell pair metadata
    results_df = pd.DataFrame(results).set_index('interaction_id')
    

    scores_matrix = np.zeros((len(results_df), len(lr_pairs)))
    
    # Fill scores matrix
    for idx, (_, row) in enumerate(df_lrp.iterrows()):
        ligand, receptor = row['ligand'], row['receptor']
        
        if ligand in adata.var_names and receptor in adata.var_names:
            for i, (_, cell_pair) in enumerate(results_df.iterrows()):
                sample_data = adata[adata.obs['sample'] == cell_pair['sample']]
                sender_expr = sample_data[sample_data.obs.index == cell_pair['sender_cell'], ligand].X
                receiver_expr = sample_data[sample_data.obs.index == cell_pair['receiver_cell'], receptor].X
                
                if issparse(sender_expr):
                    sender_expr = sender_expr.toarray().flatten()
                    receiver_expr = receiver_expr.toarray().flatten()
                
                scores_matrix[i, idx] = sender_expr[0] * receiver_expr[0]
    
    # Create AnnData object
    results_adata = anndata.AnnData(
        X=scores_matrix,
        obs=results_df,
        var=pd.DataFrame(index=lr_pairs)
    )
    
    return results_adata