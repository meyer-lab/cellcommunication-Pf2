"""
Figure A3f: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
)

def makeFigure():
    ax, f = getSetup((18, 6), (2, 3))  # 1 row, 3 columns for 3 L-R  pairs
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_covid19.h5ad")
    ccc_rise_cmp1 = 3
    ccc_rise_cmp2 = 5
    
    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]
    X_mdc_sender = add_obs_cmp_both_label(X_mdc_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2, pos1=True, pos2=True, top_perc=10, type="sender")
    X_mdc_sender = add_obs_cmp_unique_two(X_mdc_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2)
    X_mdc_sender = X_mdc_sender[X_mdc_sender.obs["Label"] != "NoLabel"]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_mdc_receiver = add_obs_cmp_both_label(X_mdc_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2, pos1=True, pos2=True, top_perc=10, type="receiver")
    X_mdc_receiver = add_obs_cmp_unique_two(X_mdc_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2)
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    print("Epithelial sender cells:", X_mdc_sender.shape)
    print("Epithelial receiver cells:", X_mdc_receiver.shape)
    
    # Calculate average communication scores for each label category
    import pandas as pd
    
    # Define ligand-receptor pairs to analyze
    pairs = [["PTN", "PTPRZ1"], ["PTN", "SDC1"], ["COL4A5", "SDC1"], ["CDH1", "CDH1"], ["OCLN", "OCLN"], ["PRSS3", "F2RL1"]]
    
    # Collect communication scores for all label combinations
    communication_data = []
    
    for lig, rec in pairs:
        pair_name = f"{lig}-{rec}"
        
        # Get expression product matrix
        df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        
        # Get sender and receiver labels
        sender_labels = X_mdc_sender.obs["Label"].values
        receiver_labels = X_mdc_receiver.obs["Label"].values
        print(sender_labels)
        print(receiver_labels)
        
        # Calculate average communication score for each sender-receiver label combination
        for sender_label in np.unique(sender_labels):
            for receiver_label in np.unique(receiver_labels):
                # Get indices for this label combination
                sender_idx = np.where(sender_labels == sender_label)[0]
                receiver_idx = np.where(receiver_labels == receiver_label)[0]
                
                if len(sender_idx) > 0 and len(receiver_idx) > 0:
                    # Extract submatrix for this label combination
                    submatrix = df.iloc[sender_idx, receiver_idx]
                    
                    # Calculate average communication score
                    avg_comm_score = submatrix.values.mean()
                    
                    communication_data.append({
                        'pair': pair_name,
                        'sender_label': sender_label,
                        'receiver_label': receiver_label,
                        'label_combination': f"{sender_label}→{receiver_label}",
                        'communication_score': avg_comm_score,
                        'n_sender_cells': len(sender_idx),
                        'n_receiver_cells': len(receiver_idx),
                        'n_interactions': len(sender_idx) * len(receiver_idx)
                    })
    
    # Convert to DataFrame
    comm_df = pd.DataFrame(communication_data)
    print("Communication data shape:", comm_df.shape)
    print("Unique pairs:", comm_df['pair'].unique())
    print("Communication data sample:")
    print(comm_df.head())
    
    # Create separate heatmap for each ligand-receptor pair
    for i, pair_name in enumerate(pairs):
        pair_label = f"{pair_name[0]}-{pair_name[1]}"
        
        # Filter data for this specific pair
        pair_data = comm_df[comm_df['pair'] == pair_label]
        
        if len(pair_data) == 0:
            print(f"No data found for pair: {pair_label}")
            continue
            
        # Create pivot table for this pair
        pivot_data = pair_data.pivot_table(
            values='communication_score', 
            index='sender_label', 
            columns='receiver_label', 
            aggfunc='mean'
        )
        
        print(f"\nPivot data for {pair_label}:")
        print(pivot_data)
        # Normalize from 0 to 1 for better visualization
        pivot_data = (pivot_data - pivot_data.min().min()) / (pivot_data.max().max() - pivot_data.min().min())

        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.4f', 
            cmap="Purples",
            cbar_kws={'label': 'Avg Communication Score'},
            ax=ax[i]
        )
        
        ax[i].set_title(f'{pair_label} Communication Scores\n(Sender → Receiver)')
        ax[i].set_xlabel('Receiver Label')
        ax[i].set_ylabel('Sender Label')
        
        # # Add text with cell counts for context
        # textstr = f'Pairs analyzed: {pair_data["label_combination"].nunique()}\n'
        # textstr += f'Total interactions: {pair_data["n_interactions"].sum():,}'
        # ax[i].text(0.02, 0.98, textstr, transform=ax[i].transAxes, 
        #           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # # Print detailed results for each pair
    # print(f"\n{'='*80}")
    # print("DETAILED COMMUNICATION SCORES BY LIGAND-RECEPTOR PAIR:")
    # print(f"{'='*80}")
    
    # for pair_name in pairs:
    #     pair_label = f"{pair_name[0]}-{pair_name[1]}"
    #     pair_data = comm_df[comm_df['pair'] == pair_label]
        
    #     if len(pair_data) > 0:
    #         print(f"\n{pair_label} Results:")
    #         print("-" * 40)
            
    #         # Show results sorted by communication score
    #         pair_summary = pair_data[['label_combination', 'communication_score', 'n_sender_cells', 'n_receiver_cells', 'n_interactions']].sort_values('communication_score', ascending=False)
    #         print(pair_summary.to_string(index=False))
            
    #         # Summary statistics
    #         print(f"\nSummary for {pair_label}:")
    #         print(f"  Highest communication: {pair_summary.iloc[0]['label_combination']} ({pair_summary.iloc[0]['communication_score']:.4f})")
    #         print(f"  Lowest communication: {pair_summary.iloc[-1]['label_combination']} ({pair_summary.iloc[-1]['communication_score']:.4f})")
    #         print(f"  Mean communication score: {pair_data['communication_score'].mean():.4f}")
    #         print(f"  Total sender-receiver combinations: {len(pair_data)}")
    #     else:
    #         print(f"\nNo data available for {pair_label}")
    
    # # Overall comparison across pairs
    # print(f"\n{'='*80}")
    # print("COMPARISON ACROSS LIGAND-RECEPTOR PAIRS:")
    # print(f"{'='*80}")
    
    # pair_comparison = comm_df.groupby('pair').agg({
    #     'communication_score': ['mean', 'std', 'min', 'max'],
    #     'label_combination': 'count'
    # }).round(4)
    
    # pair_comparison.columns = ['Mean_Score', 'Std_Score', 'Min_Score', 'Max_Score', 'N_Combinations']
    # print(pair_comparison)
    
    # # Print summary statistics
    # print(f"\n{'='*60}")
    # print("COMMUNICATION SCORE SUMMARY BY LABELS:")
    # print(f"{'='*60}")
    
    # print("\nBy Sender Label:")
    # sender_summary = sender_df.groupby('label')['communication_score'].agg(['mean', 'std', 'count'])
    # print(sender_summary)
    
    # print("\nBy Receiver Label:")
    # receiver_summary = receiver_df.groupby('label')['communication_score'].agg(['mean', 'std', 'count'])
    # print(receiver_summary)
    
    # print("\nBy Label Combination:")
    # combination_summary = comm_df.groupby('label_combination')['communication_score'].agg(['mean', 'std', 'count'])
    # print(combination_summary.sort_values('mean', ascending=False))
    
    return f





def add_obs_cmp_both_label(
    X: anndata.AnnData, cmp1: int, cmp2: int, pos1=True, pos2=True, top_perc=1, type="sender"
):
    """Adds if cells in top/bot percentage"""
    if type == "sender":
        factor_type = X.obsm["sc_B"]
    elif type == "receiver":
        factor_type = X.obsm["rc_C"]
  
    pos_neg = [pos1, pos2]

    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold1[cmp - 1]

            else:
                thres_value = top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold1[cmp - 1]

        if i == 1:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold2[cmp - 1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold2[cmp - 1]

        X.obs[f"Cmp{cmp}"] = idx

    if pos1 is True and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is True and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )

    X.obs["Both"] = idx

    return X





def add_obs_cmp_unique_two(X: anndata.AnnData, cmp1: str, cmp2: str):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False), "Label"] = "NoLabel"
    
    return X
