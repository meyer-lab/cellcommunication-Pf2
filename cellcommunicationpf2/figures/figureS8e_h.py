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
    ax, f = getSetup((10, 4), (1, 2))
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
    patient_continuous = ["age", "timeaftertx", "baselineFEV1", "FEV16monthsb4", "FEV16monthsb4p", "FEV1alad", "FEV1aladp", "FEV16monthsafter", "FEV16monthsafterp", "FEV11yearafter", "FEV1pyearafter", "d2b_index"]

    # Get unique samples with their information
    samples_df = X.obs[["dsco_id", "alad_status"] + patient_info].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)


    patient_continuous_df = X.obs[["dsco_id", "alad_status"] + patient_continuous].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    patient_info_df = X.obs[["dsco_id", "alad_status"] + patient_info].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)

    # Get the tensor decomposition factor matrix A (condition/sample factors)
    factor_A = X.uns["A"]  # Shape: (n_samples, n_components)
    n_components = factor_A.shape[1]
    
    # Create a mapping from sample to factor values
    sample_to_idx = {sample: idx for idx, sample in enumerate(X.obs["dsco_id"].cat.categories)}
    
    # Analyze all components
    target_components = [5, 13, 17]  # Specific components of interest
    target_indices = [comp - 1 for comp in target_components]  # Convert to 0-indexed
    
    # Create DataFrame combining factor values with continuous variables
    combined_data_cont = []
    for _, sample_row in patient_continuous_df.iterrows():
        sample = sample_row["dsco_id"]
        if sample in sample_to_idx:
            sample_idx = sample_to_idx[sample]
            row_data = {"sample": sample}
            for comp_idx in target_indices:
                comp_name = f"Component_{comp_idx + 1}"
                row_data[comp_name] = factor_A[sample_idx, comp_idx]
            for cont_var in patient_continuous:
                row_data[cont_var] = sample_row[cont_var]
            combined_data_cont.append(row_data)
    combined_df_cont = pd.DataFrame(combined_data_cont)

    # Pearson correlations for continuous variables
    correlation_results = []
    for comp_idx in target_indices:
        comp_name = f"Component_{comp_idx + 1}"
        for cont_var in patient_continuous:
            valid_mask = combined_df_cont[comp_name].notna() & combined_df_cont[cont_var].notna()
            valid_comp = combined_df_cont.loc[valid_mask, comp_name]
            valid_var = combined_df_cont.loc[valid_mask, cont_var]
            if len(valid_comp) > 3:
                try:
                    _, p_value = pearsonr(valid_comp, valid_var)
                    correlation_results.append({
                        "component": comp_name,
                        "variable": cont_var,
                        "p_value": p_value
                    })
                except Exception as e:
                    print(f"Pearson failed for {comp_name}, {cont_var}: {e}")
    pearson_df = pd.DataFrame(correlation_results)

    # Create DataFrame combining factor values with categorical variables
    combined_data_cat = []
    for _, sample_row in patient_info_df.iterrows():
        sample = sample_row["dsco_id"]
        if sample in sample_to_idx:
            sample_idx = sample_to_idx[sample]
            row_data = {"sample": sample}
            for comp_idx in target_indices:
                comp_name = f"Component_{comp_idx + 1}"
                row_data[comp_name] = factor_A[sample_idx, comp_idx]
            for cat_var in patient_info:
                row_data[cat_var] = sample_row[cat_var]
            combined_data_cat.append(row_data)
    combined_df_cat = pd.DataFrame(combined_data_cat)

    # T-test p-values for categorical variables
    ttest_results = []
    for comp_idx in target_indices:
        comp_name = f"Component_{comp_idx + 1}"
        for cat_var in patient_info:
            valid_mask = combined_df_cat[comp_name].notna() & combined_df_cat[cat_var].notna()
            valid_comp = combined_df_cat.loc[valid_mask, comp_name]
            valid_cat = combined_df_cat.loc[valid_mask, cat_var]
            if valid_cat.nunique() == 2 and len(valid_comp) > 3:
                try:
                    groups = [valid_comp[valid_cat == g] for g in valid_cat.unique()]
                    if all(len(g) > 1 for g in groups):
                        _, p_value = ttest_ind(groups[0], groups[1], nan_policy='omit')
                        ttest_results.append({
                            "component": comp_name,
                            "variable": cat_var,
                            "p_value": p_value
                        })
                except Exception as e:
                    print(f"T-test failed for {comp_name}, {cat_var}: {e}")
    ttest_df = pd.DataFrame(ttest_results)
    # Plot heatmaps
    if not pearson_df.empty:
        pearson_pval_matrix = pearson_df.pivot(index="component", columns="variable", values="p_value")
        sns.heatmap(pearson_pval_matrix, ax=ax[0], cmap="viridis_r", cbar_kws={'label': 'Pearson p-value'},  vmin=0, vmax=1)
        ax[0].set_xlabel("Continuous Variable")
        ax[0].set_ylabel("Component")
        ax[0].tick_params(axis='x', rotation=90)
    else:
        ax[0].text(0.5, 0.5, "No Pearson p-values", ha='center', va='center', transform=ax[0].transAxes)
        ax[0].set_title("Pearson p-value Heatmap (Continuous)")
    if not ttest_df.empty:
        ttest_pval_matrix = ttest_df.pivot(index="component", columns="variable", values="p_value")
        sns.heatmap(ttest_pval_matrix, ax=ax[1], cmap="mako_r", cbar_kws={'label': 'T-test p-value'}, vmin=0, vmax=1)
        ax[1].set_xlabel("Categorical Variable")
        ax[1].set_ylabel("Component")
        
        ax[1].tick_params(axis='x', rotation=45)
    else:
        ax[1].text(0.5, 0.5, "No T-test p-values", ha='center', va='center', transform=ax[1].transAxes)
        ax[1].set_title("T-test p-value Heatmap (Categorical)")
    # Hide unused axes
    for i in range(2, len(ax)):
        ax[i].axis('off')
    return f
