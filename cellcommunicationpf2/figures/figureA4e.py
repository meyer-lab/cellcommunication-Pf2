"""
Figure A4d: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp = 3
    
    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    X_mdc_sender = add_obs_cmp_label(X_mdc_sender, cmp=ccc_rise_cmp, pos=True, top_perc=5, type="sender")
    X_mdc_sender = add_obs_cmp_unique_one(X_mdc_sender, cmp=ccc_rise_cmp)
    # X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp-1])]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_mdc_receiver = add_obs_cmp_label(X_mdc_receiver, cmp=ccc_rise_cmp, pos=True, top_perc=5, type="receiver")
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    # X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    # X_mdc_receiver = X_mdc_receiver[np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp-1])]
    
    print("Epithelial sender cells:", X_mdc_sender.shape)

    pairs = [["PTN", "PTPRZ1"], ["PTN", "SDC1"], ["COL4A5", "SDC1"], ["MDK", "PTPRZ1"], ["MDK", "SDC1"]]
    # Want boxplot across these average communication score across 3 conditions for each of these pairs
    
    # Calculate communication scores for each pair per sample
    communication_data = []
    
    for ligand, receptor in pairs:
        pair_name = f"{ligand}-{receptor}"
        
        # Get expression product matrix between sender and receiver cells
        expr_product_df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, ligand, receptor)
        
        # For each sample, calculate average communication score
        for sample in X.obs["sample"].unique():
            # Get condition for this sample
            sample_condition = X.obs[X.obs["sample"] == sample]["condition"].iloc[0]
            
            # Get sender cells from this sample
            sender_sample_mask = X_mdc_sender.obs["sample"] == sample
            sender_cells_sample = X_mdc_sender.obs_names[sender_sample_mask]
            
            # Get receiver cells from this sample
            receiver_sample_mask = X_mdc_receiver.obs["sample"] == sample
            receiver_cells_sample = X_mdc_receiver.obs_names[receiver_sample_mask]
            
            # Extract submatrix for this sample
            if len(sender_cells_sample) > 0 and len(receiver_cells_sample) > 0:
                sample_submatrix = expr_product_df.loc[sender_cells_sample, receiver_cells_sample]
                
                # Calculate average communication score for this sample and pair for no label and labeled cells only
                avg_comm_score_no_label = sample_submatrix[sample_submatrix.index.isin(X_mdc_sender.obs_names[X_mdc_sender.obs["Label"] == "NoLabel"])].values.mean()
                avg_comm_score_label = sample_submatrix[sample_submatrix.index.isin(X_mdc_sender.obs_names[X_mdc_sender.obs["Label"] != "NoLabel"])].values.mean()

                communication_data.append({
                    'pair': pair_name,
                    'sample': sample,
                    'condition': sample_condition,
                    'communication_score_no_label': avg_comm_score_no_label,
                    'communication_score_label': avg_comm_score_label
                })
    
    # Convert to DataFrame
    comm_df = pd.DataFrame(communication_data)
    print("Communication scores data:")
    print(comm_df)
    
    # Create boxplot
    sns.boxplot(data=comm_df, x='pair', y='communication_score_label', hue='condition', ax=ax[0])
    ax[0].set_title('Average Communication Scores by L-R Pair and Condition')
    ax[0].set_xlabel('Ligand-Receptor Pair')
    ax[0].set_ylabel('Average Communication Score')
    ax[0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=comm_df, x='pair', y='communication_score_no_label', hue='condition', ax=ax[1])
    ax[1].set_title('Average Communication Scores by L-R Pair and Condition')
    ax[1].set_xlabel('Ligand-Receptor Pair')
    ax[1].set_ylabel('Average Communication Score')
    ax[1].tick_params(axis='x', rotation=45)
    
    
    # Plot both on the same plot for comparison with different names for legend with different colors for labeled vs no label. Combine into one plot and combine condition with label/no label
    comm_df_melted = pd.melt(comm_df, id_vars=['pair', 'sample', 'condition'], value_vars=['communication_score_no_label', 'communication_score_label'], var_name='label_type', value_name='communication_score')
    comm_df_melted['label_type'] = comm_df_melted['label_type'].map({
        'communication_score_no_label': 'No Label',
        'communication_score_label': 'Labeled'
    })
    comm_df_melted['condition_label'] = comm_df_melted['condition'] + ' - ' + comm_df_melted['label_type']
    sns.boxplot(data=comm_df_melted, x='pair', y='communication_score', hue='condition_label', ax=ax[2])
    ax[2].set_title('Average Communication Scores by L-R Pair, Condition, and Label Type')
    ax[2].set_xlabel('Ligand-Receptor Pair')
    ax[2].set_ylabel('Average Communication Score')
    ax[2].tick_params(axis='x', rotation=45)
    
    

    return f


def expression_product_matrix(X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str):
    """
    For each cell in X1 and each cell in X2, compute the product:
    X1[cell_i, ligand] * X2[cell_j, receptor]
    Returns a DataFrame with X1 cells as rows and X2 cells as columns.
    """
    # Ensure gene names are present
    assert ligand in X1.var_names, f"{ligand} not in X1"
    assert receptor in X2.var_names, f"{receptor} not in X2"
        
    # Get expression vectors

    # Ensure 1D dense arrays
    # Convert to dense 1D arrays, even if sparse
    expr1 = X1[:, ligand].X
    if hasattr(expr1, 'toarray'):
        expr1 = expr1.toarray().flatten()
    else:
        expr1 = np.ravel(np.array(expr1))

    expr2 = X2[:, receptor].X
    if hasattr(expr2, 'toarray'):
        expr2 = expr2.toarray().flatten()
    else:
        expr2 = np.ravel(np.array(expr2))
        
    # Compute outer product
    product_matrix = np.outer(expr1, expr2)

    # Build DataFrame
    df = pd.DataFrame(
        product_matrix,
        index=X1.obs_names,
        columns=X2.obs_names
    )
    return df