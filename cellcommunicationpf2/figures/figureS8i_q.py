"""
Figure S8i-q: Box plots of condition and ALAD status comparisons for specific components in CCC-RISE on BALF alad data.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import anndata
import pandas as pd
from .commonFuncs.plotPaCMAP import plot_wc_per_celltype, plot_wc_pacmap
from scipy.stats import pearsonr, ttest_ind, f_oneway
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def makeFigure():
    ax, f = getSetup((6, 6), (3, 3))  # 3 rows x 4 columns for 12 plots total
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    X.obs["alad_status"] = X.obs["ALADstatus"].astype(str).replace({"recovered": "ALAD", "declined": "ALAD"})
    
    # Convert d2b_index to continuous/numeric variable
    X.obs["d2b_index"] = pd.to_numeric(X.obs["d2b_index"], errors='coerce')
    print(f"d2b_index data type after conversion: {X.obs['d2b_index'].dtype}")
    print(f"d2b_index unique values: {X.obs['d2b_index'].unique()}")

    # Define patient categorical information
    patient_info = ["diagnosisgroup", "sex", "transplanttype", "cmvstatus", "cmv_status", "cmvstatus2", "ethnicity", "6monthcondition", "1yearcondition"]
    
    # Define patient continuous variables
    patient_continuous = ["percent.mt", "percent.ribo", "isotype_ctl_max", "doubletFinderScore", "S.Score", "G2M.Score", "age", "timeaftertx", "Cells (M)", "baselineFEV1", "FEV16monthsb4", "FEV16monthsb4p", "FEV1alad", "FEV1aladp", "FEV16monthsafter", "FEV16monthsafterp", "FEV11yearafter", "FEV1pyearafter", "d2b_index"]

    # Get unique samples with their information
    samples_df = X.obs[["dsco_id", "alad_status"] + patient_info].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    patient_continuous_df = X.obs[["dsco_id", "alad_status"] + patient_continuous].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    patient_info_df = X.obs[["dsco_id", "alad_status"] + patient_info].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)

    print(patient_info_df["6monthcondition"].unique())
    print(patient_info_df["1yearcondition"].unique())
    
    # Create simplified outcome categories
    def categorize_outcome(condition):
        condition_str = str(condition)
        
        # Handle exact category matches
        if condition_str == 'Died':
            return 'Died'
        elif condition_str in ['CLAD', 'CLAD/Stable', 'CLAD/Declined']:
            return 'CLAD'
        elif condition_str in ['No CLAD/Stable', 'No CLAD/Recovered', 'No CLAD/Declined', 'Recovered']:
            return 'No CLAD'

    # Apply outcome categorization to both 6-month and 1-year conditions
    patient_info_df['6monthcondition'] = patient_info_df['6monthcondition'].apply(categorize_outcome)
    patient_info_df['1yearcondition'] = patient_info_df['1yearcondition'].apply(categorize_outcome)

    # Get the tensor decomposition factor matrix A (condition/sample factors)
    factor_A = X.uns["A"]  # Shape: (n_samples, n_components)
    n_components = factor_A.shape[1]
    
    # Create a mapping from sample to factor values
    sample_to_idx = {sample: idx for idx, sample in enumerate(X.obs["dsco_id"].cat.categories)}
    
    # Analyze all components
    target_components = [5, 13, 17]  # Specific components of interest
    target_indices = [comp - 1 for comp in target_components]  # Convert to 0-indexed
    
    # Create DataFrame with factor values and sample metadata
    factor_df_list = []
    for _, sample_row in patient_info_df.iterrows():
        sample = sample_row["dsco_id"]
        if sample in sample_to_idx:
            sample_idx = sample_to_idx[sample]
            
            # Add factor values for target components only
            for comp_idx in target_indices:
                row_data = {
                    "sample": sample,
                    "alad_status": sample_row["alad_status"],
                    "component": f"Component_{comp_idx + 1}",
                    "factor_value": factor_A[sample_idx, comp_idx],
                    "component_idx": comp_idx
                }
                # Add all patient info variables
                for info_var in patient_info:
                    row_data[info_var] = sample_row[info_var]
                factor_df_list.append(row_data)
    
    factor_df = pd.DataFrame(factor_df_list)
    
    # Perform ANOVA tests for each component and condition
    anova_results = {}
    
    print("ANOVA Results:")
    print("=" * 50)
    
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        anova_results[comp_num] = {}
        
        # ANOVA for 6-month condition
        data_6m = comp_data.dropna(subset=["6monthcondition"])
        if not data_6m.empty:
            categories_6m = data_6m["6monthcondition"].unique()
            if len(categories_6m) >= 2:  # Need at least 2 groups for ANOVA
                groups_6m = []
                group_names_6m = []
                
                for cat in categories_6m:
                    group_data = data_6m[data_6m["6monthcondition"] == cat]["factor_value"].dropna()
                    if len(group_data) >= 2:  # Need at least 2 samples per group
                        groups_6m.append(group_data)
                        group_names_6m.append(cat)
                
                if len(groups_6m) >= 2:
                    try:
                        f_stat_6m, p_val_6m = f_oneway(*groups_6m)
                        anova_results[comp_num]["6month"] = {"f_stat": f_stat_6m, "p_value": p_val_6m, "groups": group_names_6m}
                        
                        # Format p-value for display
                        if p_val_6m < 0.001:
                            p_str_6m = "p < 0.001"
                        elif p_val_6m < 0.01:
                            p_str_6m = "p < 0.01"
                        elif p_val_6m < 0.05:
                            p_str_6m = f"p = {p_val_6m:.3f}*"
                        else:
                            p_str_6m = f"p = {p_val_6m:.3f}"
                        
                        print(f"Component {comp_num} - 6-month condition:")
                        print(f"  F = {f_stat_6m:.3f}, {p_str_6m}")
                        print(f"  Groups: {group_names_6m}")
                        print(f"  Group sizes: {[len(data_6m[data_6m['6monthcondition'] == cat]) for cat in group_names_6m]}")
                        
                    except Exception as e:
                        print(f"ANOVA failed for Component {comp_num}, 6-month condition: {e}")
        
        # ANOVA for 1-year condition
        data_1y = comp_data.dropna(subset=["1yearcondition"])
        if not data_1y.empty:
            categories_1y = data_1y["1yearcondition"].unique()
            if len(categories_1y) >= 2:  # Need at least 2 groups for ANOVA
                groups_1y = []
                group_names_1y = []
                
                for cat in categories_1y:
                    group_data = data_1y[data_1y["1yearcondition"] == cat]["factor_value"].dropna()
                    if len(group_data) >= 2:  # Need at least 2 samples per group
                        groups_1y.append(group_data)
                        group_names_1y.append(cat)
                
                if len(groups_1y) >= 2:
                    try:
                        f_stat_1y, p_val_1y = f_oneway(*groups_1y)
                        anova_results[comp_num]["1year"] = {"f_stat": f_stat_1y, "p_value": p_val_1y, "groups": group_names_1y}
                        
                        # Format p-value for display
                        if p_val_1y < 0.001:
                            p_str_1y = "p < 0.001"
                        elif p_val_1y < 0.01:
                            p_str_1y = "p < 0.01"
                        elif p_val_1y < 0.05:
                            p_str_1y = f"p = {p_val_1y:.3f}*"
                        else:
                            p_str_1y = f"p = {p_val_1y:.3f}"
                        
                        print(f"Component {comp_num} - 1-year condition:")
                        print(f"  F = {f_stat_1y:.3f}, {p_str_1y}")
                        print(f"  Groups: {group_names_1y}")
                        print(f"  Group sizes: {[len(data_1y[data_1y['1yearcondition'] == cat]) for cat in group_names_1y]}")
                        
                    except Exception as e:
                        print(f"ANOVA failed for Component {comp_num}, 1-year condition: {e}")
        
        print()  # Add blank line between components
    
    # Perform t-tests for ALAD status comparisons
    ttest_results = {}
    
    print("T-Test Results for ALAD Status:")
    print("=" * 40)
    
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        
        # Get data for each ALAD status group
        alad_groups = comp_data["alad_status"].unique()
        
        if len(alad_groups) == 2:  # Need exactly 2 groups for t-test
            group1_name = alad_groups[0]
            group2_name = alad_groups[1]
            
            group1_data = comp_data[comp_data["alad_status"] == group1_name]["factor_value"].dropna()
            group2_data = comp_data[comp_data["alad_status"] == group2_name]["factor_value"].dropna()
            
            if len(group1_data) >= 2 and len(group2_data) >= 2:
                try:
                    t_stat, p_val = ttest_ind(group1_data, group2_data)
                    ttest_results[comp_num] = {
                        "t_stat": t_stat, 
                        "p_value": p_val, 
                        "group1": group1_name, 
                        "group2": group2_name,
                        "n1": len(group1_data),
                        "n2": len(group2_data)
                    }
                    
                    # Format p-value for display
                    if p_val < 0.001:
                        p_str = "p < 0.001"
                    elif p_val < 0.01:
                        p_str = "p < 0.01"
                    elif p_val < 0.05:
                        p_str = f"p = {p_val:.3f}*"
                    else:
                        p_str = f"p = {p_val:.3f}"
                    
                    print(f"Component {comp_num} - ALAD Status:")
                    print(f"  t = {t_stat:.3f}, {p_str}")
                    print(f"  Groups: {group1_name} (n={len(group1_data)}) vs {group2_name} (n={len(group2_data)})")
                    
                except Exception as e:
                    print(f"T-test failed for Component {comp_num}, ALAD status: {e}")
        
        print()  # Add blank line between components
    
    # Create plots in organized layout
    # Row 1: 6-month condition (positions 0, 1, 2)
    # Row 2: 1-year condition (positions 3, 4, 5) 
    # Row 3: ALAD status (positions 6, 7, 8)
    
    for i, comp_num in enumerate(target_components):
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        
        # Plot 1: 6-month condition (top row)
        plot_idx_6m = i  # positions 0, 1, 2
        plot_data_6m = comp_data.dropna(subset=["6monthcondition"])
        
        if not plot_data_6m.empty:
            sns.boxplot(data=plot_data_6m, x="6monthcondition", y="factor_value", ax=ax[plot_idx_6m])
            
            # Add ANOVA results to title if available
            title = f"Component {comp_num}\n6-Month Condition"
            if comp_num in anova_results and "6month" in anova_results[comp_num]:
                f_stat = anova_results[comp_num]["6month"]["f_stat"]
                p_val = anova_results[comp_num]["6month"]["p_value"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                elif p_val < 0.05:
                    p_str = f"p = {p_val:.3f}*"
                else:
                    p_str = f"p = {p_val:.3f}"
                title += f"\nF = {f_stat:.2f}, {p_str}"
            
            ax[plot_idx_6m].set_title(title, fontsize=10)
            ax[plot_idx_6m].set_xlabel("6-Month Condition")
            ax[plot_idx_6m].set_ylabel("Factor Value")
            
            # Add sample size information
            total_n = len(plot_data_6m)
            ax[plot_idx_6m].text(0.02, 0.98, f"n = {total_n}", transform=ax[plot_idx_6m].transAxes, 
                             verticalalignment='top', fontsize=8)
            
            # Rotate x-axis labels
            plt.setp(ax[plot_idx_6m].get_xticklabels(), rotation=45, ha='right')
        else:
            ax[plot_idx_6m].text(0.5, 0.5, "No data available", ha='center', va='center', 
                             transform=ax[plot_idx_6m].transAxes)
            ax[plot_idx_6m].set_title(f"Component {comp_num} - 6-Month Condition")
        
        # Plot 2: 1-year condition (middle row)
        plot_idx_1y = i + 3  # positions 3, 4, 5
        plot_data_1y = comp_data.dropna(subset=["1yearcondition"])
        
        if not plot_data_1y.empty:
            sns.boxplot(data=plot_data_1y, x="1yearcondition", y="factor_value", ax=ax[plot_idx_1y])
            
            # Add ANOVA results to title if available
            title = f"Component {comp_num}\n1-Year Condition"
            if comp_num in anova_results and "1year" in anova_results[comp_num]:
                f_stat = anova_results[comp_num]["1year"]["f_stat"]
                p_val = anova_results[comp_num]["1year"]["p_value"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                elif p_val < 0.05:
                    p_str = f"p = {p_val:.3f}*"
                else:
                    p_str = f"p = {p_val:.3f}"
                title += f"\nF = {f_stat:.2f}, {p_str}"
            
            ax[plot_idx_1y].set_title(title, fontsize=10)
            ax[plot_idx_1y].set_xlabel("1-Year Condition")
            ax[plot_idx_1y].set_ylabel("Factor Value")
            
            # Add sample size information
            total_n = len(plot_data_1y)
            ax[plot_idx_1y].text(0.02, 0.98, f"n = {total_n}", transform=ax[plot_idx_1y].transAxes, 
                             verticalalignment='top', fontsize=8)
            
            # Rotate x-axis labels
            plt.setp(ax[plot_idx_1y].get_xticklabels(), rotation=45, ha='right')
        else:
            ax[plot_idx_1y].text(0.5, 0.5, "No data available", ha='center', va='center', 
                             transform=ax[plot_idx_1y].transAxes)
            ax[plot_idx_1y].set_title(f"Component {comp_num} - 1-Year Condition")
        
        # Plot 3: ALAD status (bottom row)
        plot_idx_alad = i + 6  # positions 6, 7, 8
        
        sns.boxplot(data=comp_data, x="alad_status", y="factor_value", ax=ax[plot_idx_alad])
        
        # Add t-test results to title if available
        title = f"Component {comp_num}\nALAD Status"
        if comp_num in ttest_results:
            t_stat = ttest_results[comp_num]["t_stat"]
            p_val = ttest_results[comp_num]["p_value"]
            if p_val < 0.001:
                p_str = "p < 0.001"
            elif p_val < 0.01:
                p_str = "p < 0.01"
            elif p_val < 0.05:
                p_str = f"p = {p_val:.3f}*"
            else:
                p_str = f"p = {p_val:.3f}"
            title += f"\nt = {t_stat:.2f}, {p_str}"
        
        ax[plot_idx_alad].set_title(title, fontsize=10)
        ax[plot_idx_alad].set_xlabel("ALAD Status")
        ax[plot_idx_alad].set_ylabel("Factor Value")
        
        # Add sample size information
        total_n = len(comp_data)
        ax[plot_idx_alad].text(0.02, 0.98, f"n = {total_n}", transform=ax[plot_idx_alad].transAxes, 
                         verticalalignment='top', fontsize=8)
    
    # All 9 positions should be used now, no need to hide any axes
    
    # Print summary statistics
    print("Summary of condition categories:")
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        print(f"\nComponent {comp_num}:")
        
        # 6-month condition stats
        data_6m = comp_data.dropna(subset=["6monthcondition"])
        if not data_6m.empty:
            print(f"  6-month condition counts: {data_6m['6monthcondition'].value_counts().to_dict()}")
        
        # 1-year condition stats
        data_1y = comp_data.dropna(subset=["1yearcondition"])
        if not data_1y.empty:
            print(f"  1-year condition counts: {data_1y['1yearcondition'].value_counts().to_dict()}")
    
    return f