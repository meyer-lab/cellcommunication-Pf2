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
    X = X[::50, :10000]
    print(X)
  
            
            

    

    
    XX = calculate_communication_scores_simple(X, df_lrp)
    print(XX)

    
    

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
    Simplified version to calculate cell-cell communication scores.
    
    Parameters:
    - adata: AnnData object
    - df_lrp: DataFrame with ligand-receptor pairs
    - min_expr: Minimum expression threshold (default 0.0)
    - same_cell: Whether to include same-cell interactions (default False)
    """
    # Gene matching statistics
    matched_ligands = sum(lig in adata.var_names for lig in df_lrp['ligand'].unique())
    matched_receptors = sum(rec in adata.var_names for rec in df_lrp['receptor'].unique())
    
    print(f"Matched ligands: {matched_ligands}/{len(df_lrp['ligand'].unique())}")
    print(f"Matched receptors: {matched_receptors}/{len(df_lrp['receptor'].unique())}")
    
    results = []
    
    for sample in adata.obs["sample"].unique():
        sample_data = adata[adata.obs["sample"] == sample]
        X_sparse = sample_data.X if sp.issparse(sample_data.X) else sp.csr_matrix(sample_data.X)
        
        # Find expressed genes
        expressed_genes = sample_data.var_names[(X_sparse > min_expr).sum(axis=0).A1 > 0]
        valid_pairs = [(lig, rec) for lig, rec in df_lrp[['ligand', 'receptor']].values 
                       if lig in expressed_genes and rec in expressed_genes]
        
        if not valid_pairs:
            print(f"No valid pairs in sample {sample}")
            continue
        
        # Generate cell pairs
        cell_pairs = [(s, r) for s in range(sample_data.shape[0]) for r in range(sample_data.shape[0]) 
                      if s != r or same_cell]
        
        # Calculate scores
        scores = sp.lil_matrix((len(cell_pairs), len(valid_pairs)))
        for idx, (lig, rec) in enumerate(valid_pairs):
            lig_idx = sample_data.var_names.get_loc(lig)
            rec_idx = sample_data.var_names.get_loc(rec)
            
            # Get expression values as dense arrays
            lig_vec = X_sparse[:, lig_idx].toarray().flatten()
            rec_vec = X_sparse[:, rec_idx].toarray().flatten()
            
            for pair_idx, (s, r) in enumerate(cell_pairs):
                if lig_vec[s] > min_expr and rec_vec[r] > min_expr:
                    scores[pair_idx, idx] = float(lig_vec[s] * rec_vec[r])
        
        # Prepare results
        for pair_idx, (s, r) in enumerate(cell_pairs):
            pair_scores = scores[pair_idx, :].toarray().flatten()
            if pair_scores.sum() > 0:
                results.append({
                    'sample': sample,
                    'sender': sample_data.obs.index[s],
                    'receiver': sample_data.obs.index[r],
                    **{f'score_{lig}_{rec}': score for (lig, rec), score in zip(valid_pairs, pair_scores) if score > 0}
                })
    
    if not results:
        print("No interactions found")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure all ligand-receptor pairs are included in the var DataFrame
    all_pairs = [f"{lig}_{rec}" for lig, rec in df_lrp[['ligand', 'receptor']].values]
    var_df = pd.DataFrame(index=all_pairs)
    
    # Create scores matrix with consistent columns
    score_columns = [f"score_{lig}_{rec}" for lig, rec in df_lrp[['ligand', 'receptor']].values]
    scores_matrix = np.zeros((len(results_df), len(score_columns)))
    
    for idx, col in enumerate(score_columns):
        if col in results_df.columns:
            scores_matrix[:, idx] = results_df[col].fillna(0).values
    
    # Create AnnData object
    results_adata = anndata.AnnData(
        X=sp.csr_matrix(scores_matrix),
        obs=results_df.drop(columns=results_df.filter(like='score_').columns),
        var=var_df
    )
    
    return results_adata