"""
Figure A2: Non-negative CP for COVID-19 pseudobulk
"""

import numpy as np
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from .common import (
    subplotLabel,
    getSetup,
)
from ..utils import (
    pseudobulk_X
)
from ..cc_pf2 import (
    calc_communication_score_pseudobulk,
    pseudobulk_nncp_decomposition,
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors
)

from .commonFuncs.plotGeneral import (
    rotate_yaxis
)

import pandas as pd

def makeFigure():
    ax, f = getSetup((24, 12), (2, 4))
    subplotLabel(ax)
    
    print(import_balf_covid(gene_threshold=0))
    




    # Need to compare two differnet csv of which genes overlap in the pseudobulk analysis for each condition and cell type
    # This will help to determine which genes are consistently expressed across conditions and cell types

    # Compare the two CSV files and find overlapping genes
    # This can be done using pandas to read the CSV files and find intersections
    

    # df1 = pd.read_csv("C51.csv")
    # df2 = pd.read_csv("pseudobulk_C51_celltype_fraction.csv")
    
    df1 = pd.read_csv("C145.csv")
    df2 = pd.read_csv("C145_pseudobulk.csv")
    # df2 = df2.T
    # print(df1)
    
    print(df1)
    
    print(df2)

    # Assuming the gene names are in a column (let's check the column names)
  

    # If gene names are in a specific column (e.g., 'gene', 'Gene', or the first column)
    # Let's assume the gene column is the first column in both files
    gene_col_df1 = df1.columns[0]  # Get the name of the first column
    gene_col_df2 = df2.columns[0]  # Get the name of the first column

    df1[gene_col_df1] = df1[gene_col_df1].str.upper()
    df2[gene_col_df2] = df2[gene_col_df2].str.upper()

    print(f"\nGene column in df1: {gene_col_df1}")
    print(f"Gene column in df2: {gene_col_df2}")

    # Get the gene lists from both dataframes
    genes_df1 = set(df1[gene_col_df1].dropna().unique())
    genes_df2 = set(df2[gene_col_df2].dropna().unique())

    print(f"\nNumber of genes in df1: {len(genes_df1)}")
    print(f"Number of genes in df2: {len(genes_df2)}")

    # Find overlapping genes
    overlapping_genes = genes_df1.intersection(genes_df2)
    print(f"Number of overlapping genes: {len(overlapping_genes)}")
    
    gene_col = df1.columns[0]
    common_genes = sorted(set(df1[gene_col]) & set(df2[gene_col]))

    df1_aligned = df1.set_index(gene_col).reindex(common_genes).fillna(0)
    df2_aligned = df2.set_index(gene_col).reindex(common_genes).fillna(0)

    # Ensure both dataframes have the same columns (if they should)
    # If they have different columns, you might want to compare specific ones

    print("Comparison of values:")

    # Compare overall similarity
    if df1_aligned.shape[1] == df2_aligned.shape[1]:
        # If they have the same number of columns
        for i, col in enumerate(df1_aligned.columns):
            if col in df2_aligned.columns:
                values1 = df1_aligned[col].values
                values2 = df2_aligned[col].values
                
                # Calculate similarity metrics
                correlation = np.corrcoef(values1, values2)[0, 1]
                mse = np.mean((values1 - values2) ** 2)
                rmse = np.sqrt(mse)
                
                print(f"Column {col}:")
                print(f"  Pearson correlation: {correlation:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  Mean values - df1: {np.mean(values1):.4f}, df2: {np.mean(values2):.4f}")

    # Compare specific cell types if known
    cell_types = ['Epithelial', 'Macrophages', 'NK', 'Plasma', 'T', 'mDC', 'pDC']
    for cell_type in cell_types:
        if cell_type in df1_aligned.columns and cell_type in df2_aligned.columns:
            v1 = df1_aligned[cell_type].values
            v2 = df2_aligned[cell_type].values
            corr = np.corrcoef(v1, v2)[0, 1]
            print(f"{cell_type}: correlation = {corr:.4f}")
        
        
    genes_only_in_df1 = genes_df1 - genes_df2
    print("Genes in df1 but not in df2:", genes_only_in_df1)
    genes_c5 = df2[df2[gene_col_df2].str.startswith("C5")]
    print(genes_c5)



    return f