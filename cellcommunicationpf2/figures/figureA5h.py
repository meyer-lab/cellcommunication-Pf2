"""
Figure A5h: Boxplots of associations between CCC-RISE components and patient clinical variables in BALF ALAD data.
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
# from ..utils import rotate_xaxis, rotate_yaxis



def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    X.obs["alad_status"] = X.obs["ALADstatus"].astype(str).replace({"recovered": "ALAD", "declined": "ALAD"})
    
    # Convert d2b_index to continuous/numeric variable
    X.obs["d2b_index"] = pd.to_numeric(X.obs["d2b_index"], errors='coerce')
    print(f"d2b_index data type after conversion: {X.obs['d2b_index'].dtype}")
    print(f"d2b_index unique values: {X.obs['d2b_index'].unique()}")
    print(f"d2b_index value counts:\n{X.obs['d2b_index'].value_counts()}")

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

    alad_statuses = X.obs["alad_status"].unique()
    
    # Make a t-test comparison for X component values between alad statuses for each categorical variable
    # Get the tensor decomposition factor matrix A (condition/sample factors)
    factor_A = X.uns["A"]  # Shape: (n_samples, n_components)
    n_components = factor_A.shape[1]
    
    # Create a mapping from sample to factor values
    sample_to_idx = {sample: idx for idx, sample in enumerate(X.obs["dsco_id"].cat.categories)}
    
    # Only analyze components 12, 16, and 4
    target_components = [5, 13, 17]  # 1-indexed component numbers
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
    
    # Perform t-tests for specific components only (12, 16, and 4)
    results = []
    
    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]
        
        for var in patient_info:
            # Get all categories for this variable
            var_categories = comp_data[var].dropna().unique()
            if len(var_categories) < 2:
                continue
            
            # Perform pairwise t-tests between all categories
            from itertools import combinations
            for cat1, cat2 in combinations(var_categories, 2):
                # Get factor values for each category
                group1_data = comp_data[comp_data[var] == cat1]["factor_value"].dropna()
                group2_data = comp_data[comp_data[var] == cat2]["factor_value"].dropna()
                
                # Perform t-test if both groups have sufficient data
                if len(group1_data) > 1 and len(group2_data) > 1:
                    try:
                        t_stat, p_value = ttest_ind(group1_data, group2_data)
                        results.append({
                            "component": f"Component_{comp_idx + 1}",
                            "variable": var,
                            "category1": cat1,
                            "category2": cat2,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "n_group1": len(group1_data),
                            "n_group2": len(group2_data),
                            "group1_name": cat1,
                            "group2_name": cat2
                        })
                    except Exception as e:
                        print(f"T-test failed for Component {comp_idx + 1}, {var}, {cat1} vs {cat2}: {e}")
    
    # Create results dataframe and filter for significant results (p < 0.05)
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Perform ANOVA for 3-category outcome variables (6monthcondition and 1yearcondition)
    anova_results = []
    outcome_vars = ["6monthcondition", "1yearcondition"]
    
    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]
        
        for var in outcome_vars:
            # Get data for each of the 3 categories
            var_data = comp_data.dropna(subset=[var])
            categories = var_data[var].unique()
            
            if len(categories) == 3:  # Only proceed if we have all 3 categories
                groups = []
                group_names = []
                group_sizes = []
                
                for cat in categories:
                    group_data = var_data[var_data[var] == cat]["factor_value"].dropna()
                    if len(group_data) >= 2:  # Need at least 2 samples per group
                        groups.append(group_data)
                        group_names.append(cat)
                        group_sizes.append(len(group_data))
                
                if len(groups) == 3:  # Proceed with ANOVA if all 3 groups have sufficient data
                    try:
                        f_stat, p_value = f_oneway(*groups)
                        anova_results.append({
                            "component": f"Component_{comp_idx + 1}",
                            "variable": var,
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "groups": group_names,
                            "group_sizes": group_sizes,
                            "total_n": sum(group_sizes)
                        })
                        print(f"ANOVA - Component {comp_idx + 1}, {var}: F={f_stat:.3f}, p={p_value:.4f}, groups={group_names}, n={group_sizes}")
                    except Exception as e:
                        print(f"ANOVA failed for Component {comp_idx + 1}, {var}: {e}")
    
    # Print ANOVA results
    anova_df = pd.DataFrame(anova_results)
    if not anova_df.empty:
        significant_anova = anova_df[anova_df["p_value"] < 0.05]
        print(f"\nFound {len(significant_anova)} significant ANOVA results (p < 0.05):")
        for _, row in significant_anova.iterrows():
            print(f"{row['component']} - {row['variable']}: F={row['f_statistic']:.3f}, p={row['p_value']:.4f}, n={row['total_n']}")
    
    # Combine t-test and ANOVA results for plotting
    all_significant_results = []
    
    # Add significant t-test results
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05].copy()
        significant_results = significant_results.sort_values("p_value")
        
        print(f"Found {len(significant_results)} significant t-test results (p < 0.05):")
        print(significant_results[["component", "variable", "category1", "category2", "p_value"]])
        
        # Add t-test results to plotting list
        for _, row in significant_results.iterrows():
            all_significant_results.append({
                "type": "t-test",
                "data": row,
                "plot_type": "boxplot_pairwise"
            })
    
    # Add significant ANOVA results (or top results if none are significant)
    if not anova_df.empty:
        significant_anova = anova_df[anova_df["p_value"] < 0.05]
        if len(significant_anova) == 0:
            # If no significant ANOVA results, take the top 3 with lowest p-values
            print("No significant ANOVA results, showing top 3 with lowest p-values")
            significant_anova = anova_df.nsmallest(3, "p_value")
        
        for _, row in significant_anova.iterrows():
            all_significant_results.append({
                "type": "anova",
                "data": row,
                "plot_type": "boxplot_all_groups"
            })
    
    if all_significant_results:
        # Plot up to the number of significant results found, or the number of available axes
        n_plots = min(len(all_significant_results), 9)  # Maximum 9 plots for 3x3 grid
        
        for i in range(n_plots):
            result = all_significant_results[i]
            
            if result["type"] == "t-test":
                # Handle t-test plotting (existing logic)
                row = result["data"]
                
                # Get data for this specific test
                comp_idx = int(row["component"].split("_")[1]) - 1
                var = row["variable"]
                cat1 = row["category1"]
                cat2 = row["category2"]
                
                # Filter data for the specific component and include both categories
                plot_data = factor_df[
                    (factor_df["component_idx"] == comp_idx) & 
                    (factor_df[var].isin([cat1, cat2]))
                ]
                
                # Create boxplot comparing the two categories
                sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax[i])
                
                # Calculate sample sizes for each category
                n_cat1 = len(plot_data[plot_data[var] == cat1])
                n_cat2 = len(plot_data[plot_data[var] == cat2])
                n_total = len(plot_data)
                
                # Format title with p-value and sample size
                p_val = row["p_value"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                else:
                    p_str = f"p = {p_val:.3f}"
                
                title = f"{row['component']}\n{var}: {cat1} vs {cat2}\n{p_str} (n={n_cat1}+{n_cat2}={n_total})"
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(var)
                ax[i].set_ylabel("Factor Value")
                
            elif result["type"] == "anova":
                # Handle ANOVA plotting (all 3 groups)
                row = result["data"]
                
                # Get data for this ANOVA test
                comp_idx = int(row["component"].split("_")[1]) - 1
                var = row["variable"]
                
                # Filter data for the specific component
                plot_data = factor_df[
                    (factor_df["component_idx"] == comp_idx) & 
                    (factor_df[var].notna())
                ]
                
                # Create boxplot for all categories
                sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax[i])
                
                # Format title with F-statistic and p-value
                p_val = row["p_value"]
                f_stat = row["f_statistic"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                else:
                    p_str = f"p = {p_val:.3f}"
                
                title = f"{row['component']}\n{var} (ANOVA)\nF={f_stat:.2f}, {p_str} (n={row['total_n']})"
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(var)
                ax[i].set_ylabel("Factor Value")
            
            # Rotate x-axis labels if needed
            plt.setp(ax[i].get_xticklabels(), rotation=45, ha='right')
        
        # Hide unused axes
        for i in range(n_plots, len(ax)):
            ax[i].set_visible(False)
            
    else:
        print("No significant results found (p < 0.05)")
        # Add empty plots with informative titles for first few axes
        for i in range(min(4, len(ax))):
            ax[i].text(0.5, 0.5, "No significant\nresults found", 
                      ha='center', va='center', transform=ax[i].transAxes)
            ax[i].set_title(f"Component Analysis")
        
        # Hide remaining axes
        for i in range(min(4, len(ax)), len(ax)):
            ax[i].set_visible(False)

    return f
