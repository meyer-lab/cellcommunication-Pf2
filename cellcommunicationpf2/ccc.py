import pandas as pd
import numpy as np
import anndata
from scipy.sparse import sp


def calculate_communication_scores_simple(X: anndata.AnnData, df_lrp: pd.DataFrame):
    """
    Calculate cell-cell communication scores for specific L-R pairs in AnnData.
    - adata: AnnData object with filtered genes
    - df_lrp: DataFrame with filtered ligand-receptor pairs
    """
    results = []
    print(f"Processing {len(df_lrp)} L-R pairs")
    
    for sample in np.unique(X.obs["sample_new"]):
        print(f"\nProcessing sample: {sample}")
        sample_data = X[X.obs["sample_new"] == sample]
        X_sparse = sample_data.X if sp.issparse(sample_data.X) else sp.csr_matrix(sample_data.X)
        
        # Generate cell pairs
        cell_pairs = [(s, r) for s in range(sample_data.shape[0]) 
                      for r in range(sample_data.shape[0]) 
                      if s != r]
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

            scores[:, idx] = pair_scores
        
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
    

    df_scores = pd.DataFrame(results)
    score_columns = [f'{row.ligand}_{row.receptor}' for _, row in df_lrp.iterrows()]
    
    X_ccc = anndata.AnnData(
        X=sp.csr_matrix(df_scores[score_columns].values),
        obs=df_scores.drop(columns=score_columns).set_index('interaction_id'),
        var=pd.DataFrame(index=score_columns)
    )
    
    return X_ccc