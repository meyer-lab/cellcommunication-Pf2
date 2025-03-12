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