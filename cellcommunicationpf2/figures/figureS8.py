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
from scipy.stats import pearsonr 
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
# from ..utils import rotate_xaxis, rotate_yaxis



def makeFigure():
    ax, f = getSetup((12, 12), (2, 2))
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

    print(X.obs["d2b_index"].unique)
    print(f"Sample dataframe shape: {samples_df.shape}")
    print(samples_df)
    
    for i in patient_continuous:
        print(f"Processing continuous variable: {i}")
        print(X.obs[i].dtype)

    patient_continuous_df = X.obs[["dsco_id", "alad_status"] + patient_continuous].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    
    
    # Calculate correlation between continuous variables by alad status from patient_continuous_df
    alad_statuses = X.obs["alad_status"].unique()
    

    for i, status in enumerate(alad_statuses):
        status_df = patient_continuous_df[patient_continuous_df["alad_status"] == status]
        print(f"\nProcessing {status} status:")
        print(f"Shape: {status_df.shape}")
        print("Columns:", list(status_df.columns))
        
        # Drop non-numeric columns for correlation analysis
        numeric_df = status_df.drop(columns=["dsco_id", "alad_status"])
        

            
        corr_df = correlation_df(numeric_df)
        plot_correlation_heatmap(
                corr_df,
                xticks=corr_df.columns,
                yticks=corr_df.index,
                mask=True,
                ax=ax[i])

        

    return f


def correlation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the correlation"""
    # First, convert all columns to numeric, coercing errors to NaN
    numeric_df = df.copy()
    
    # Convert each column to numeric
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        print(f"Column {col}: dtype={numeric_df[col].dtype}, non-null count={numeric_df[col].notna().sum()}")
    
    # Only keep columns that have sufficient numeric data
    valid_columns = []
    for col in numeric_df.columns:
        valid_count = numeric_df[col].notna().sum()
        if valid_count > 2:  # Need at least 3 valid values
            valid_columns.append(col)
        else:
            print(f"Skipping column {col}: only {valid_count} valid values")
    
    if len(valid_columns) < 2:
        print("Not enough valid columns for correlation analysis")
        return pd.DataFrame()
    
    # Use only valid columns
    numeric_df = numeric_df[valid_columns]
    
    pearson_df = pd.DataFrame(
        columns=valid_columns,
        index=valid_columns,
        dtype=float
    )

    for row in valid_columns:
        for column in valid_columns:
            # Get data for these two columns and drop NaN
            two_df = numeric_df[[row, column]].dropna()
            
            if len(two_df) > 2:
                try:
                    # Ensure data is actually numeric
                    row_data = two_df[row].astype(float)
                    col_data = two_df[column].astype(float)
                    
                    # Check for variance (avoid constant columns)
                    if row_data.var() > 1e-10 and col_data.var() > 1e-10:
                        result = pearsonr(row_data.values, col_data.values)
                        pearson_df.loc[row, column] = result.pvalue
                    else:
                        pearson_df.loc[row, column] = np.nan
                except (ValueError, TypeError) as e:
                    print(f"Error computing correlation between {row} and {column}: {e}")
                    pearson_df.loc[row, column] = np.nan
            else:
                pearson_df.loc[row, column] = np.nan
                
    return pearson_df



def plot_correlation_heatmap(correlation_df: pd.DataFrame, xticks, yticks, ax: Axes, mask=None):
    """Plots a heatmap of the correlation matrix"""
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    
    if mask is not None: 
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        for i in range(len(mask)):
            mask[i, i] = False
        
    sns.heatmap(
        data=correlation_df.to_numpy(),
        vmin=0,
        vmax=.05,
        xticklabels=xticks,
        yticklabels=yticks,
        mask=mask,
        cmap=cmap,
        cbar_kws={"label": "Pearson Correlation P-value"},
        ax=ax,
    )

    # rotate_xaxis(ax, rotation=90)
    # rotate_yaxis(ax, rotation=0)
