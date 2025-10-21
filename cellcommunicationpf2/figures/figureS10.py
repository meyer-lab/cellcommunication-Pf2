"""
Figure S3: CCC-RISE on BALF alad data. Showing weighted sender and receiver cell factors.
"""

from .common import (
    subplotLabel,
    getSetup,
)
import anndata
import pandas as pd
from .commonFuncs.plotPaCMAP import plot_wc_per_celltype, plot_wc_pacmap
from scipy.stats import pearsonr, ttest_ind
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
# from ..utils import rotate_xaxis, rotate_yaxis



def makeFigure():
    ax, f = getSetup((8, 8), (4, 4))
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

    
    
    # Calculate Pearson correlations between components and continuous variables
    # Get the tensor decomposition factor matrix A (condition/sample factors)
    factor_A = X.uns["A"]  # Shape: (n_samples, n_components)
    n_components = factor_A.shape[1]
    
    # Create a mapping from sample to factor values
    sample_to_idx = {sample: idx for idx, sample in enumerate(X.obs["dsco_id"].cat.categories)}
    
    # Target components for analysis
    target_components = [4, 12, 16]  # 1-indexed component numbers
    target_indices = [comp - 1 for comp in target_components]  # Convert to 0-indexed
    
    # Create DataFrame combining factor values with continuous variables
    combined_data = []
    for _, sample_row in patient_continuous_df.iterrows():
        sample = sample_row["dsco_id"]
        if sample in sample_to_idx:
            sample_idx = sample_to_idx[sample]
            
            row_data = {"sample": sample}
            
            # Add factor values for target components
            for comp_idx in target_indices:
                comp_name = f"Component_{comp_idx + 1}"
                row_data[comp_name] = factor_A[sample_idx, comp_idx]
            
            # Add continuous variables
            for cont_var in patient_continuous:
                row_data[cont_var] = sample_row[cont_var]
                
            combined_data.append(row_data)
    
    combined_df = pd.DataFrame(combined_data)
    
    
    # Calculate Pearson correlations between components and continuous variables
    correlation_results = []
    
    for comp_idx in target_indices:
        comp_name = f"Component_{comp_idx + 1}"
        comp_values = combined_df[comp_name].dropna()
        
        for cont_var in patient_continuous:
            # Get corresponding continuous variable values (aligned by sample)
            var_values = combined_df[cont_var].dropna()
            
            # Find common samples (no NaN in either component or variable)
            valid_mask = combined_df[comp_name].notna() & combined_df[cont_var].notna()
            valid_comp = combined_df.loc[valid_mask, comp_name]
            valid_var = combined_df.loc[valid_mask, cont_var]
            
            # Calculate correlation if we have sufficient data points
            if len(valid_comp) > 3:  # Need at least 4 points for meaningful correlation
                try:
                    corr_coef, p_value = pearsonr(valid_comp, valid_var)
                    correlation_results.append({
                        "component": comp_name,
                        "variable": cont_var,
                        "correlation": corr_coef,
                        "p_value": p_value,
                        "abs_correlation": abs(corr_coef),
                        "n_samples": len(valid_comp)
                    })
                except Exception as e:
                    print(f"Correlation failed for {comp_name}, {cont_var}: {e}")
    
    # Create results dataframe and sort by absolute correlation strength
    correlation_df = pd.DataFrame(correlation_results)
    if not correlation_df.empty:
        # Sort by absolute correlation (strongest correlations first)
        sorted_correlations = correlation_df.sort_values("abs_correlation", ascending=False)
        
        # Show top correlations
        print("Top correlations (sorted by absolute correlation strength):")
        print(sorted_correlations[["component", "variable", "correlation", "p_value"]].head(10))
        
        # Plot the top correlations (up to 9 for 3x3 layout)
        n_plots = min(9, len(sorted_correlations))
        print(sorted_correlations)
        for i in range(10):
            row = sorted_correlations.iloc[i]
            comp_name = row["component"]
            var_name = row["variable"]
            corr_val = row["correlation"]
            p_val = row["p_value"]
            
            # Get data for scatter plot
            valid_mask = combined_df[comp_name].notna() & combined_df[var_name].notna()
            x_data = combined_df.loc[valid_mask, comp_name]
            y_data = combined_df.loc[valid_mask, var_name]
            n_samples = len(x_data)
            
            # Create scatter plot with regression line
            sns.scatterplot(x=x_data, y=y_data, ax=ax[i], alpha=0.7)
            sns.regplot(x=x_data, y=y_data, ax=ax[i], scatter=False, color='red', line_kws={'linewidth': 2})
            
            # Format title with correlation, p-value, and sample size
            if p_val < 0.001:
                p_str = "p < 0.001"
            elif p_val < 0.01:
                p_str = "p < 0.01"
            elif p_val < 0.05:
                p_str = "p < 0.05"
            else:
                p_str = f"p = {p_val:.3f}"
            
            title = f"{comp_name} vs {var_name}\nr = {corr_val:.3f}, {p_str} (n={n_samples})"
            ax[i].set_title(title, fontsize=10)
            ax[i].set_xlabel(comp_name)
            ax[i].set_ylabel(var_name)
 
            ax[i].set_ylabel(var_name, rotation=90, ha='center')
    else:
        print("No correlation results found")
        # Add empty plots with informative titles
        for i in range(9):
            ax[i].text(0.5, 0.5, "No correlation\nresults found", 
                      ha='center', va='center', transform=ax[i].transAxes)
            ax[i].set_title(f"Plot {i+1}")

    return f

