"""
Figure A4f: Reading Seurat RDS files and RISE decomposition visualization.
"""

import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
)
from ..tensor import rise_store_r2x
from ..import_data import import_alad
from parafac2.parafac2 import parafac2_nd, anndata_to_list
from pacmap import PaCMAP

# Direct RDS reading function to check original data
def check_rds_column_directly(file_path, column_name="revised_annotations_10_5"):
    """
    Create an R script to directly check the original RDS column values
    """
    print(f"\n=== CREATING R SCRIPT TO CHECK ORIGINAL RDS FILE ===")
    print(f"File: {file_path}")
    print(f"Column: {column_name}")
    
    # Create R script to check the original data
    r_script_path = "/tmp/check_rds_column.R"
    r_script_content = f'''
# Load the RDS file
cat("Loading RDS file: {file_path}\\n")
obj <- readRDS("{file_path}")

# Check if the column exists
cat("Checking for column: {column_name}\\n")
if ("{column_name}" %in% colnames(obj@meta.data)) {{
    cat("Column found!\\n")
    
    # Get the column data
    col_data <- obj@meta.data${column_name}
    
    cat("Data type:", class(col_data), "\\n")
    cat("Length:", length(col_data), "\\n")
    
    # Print first 10 values
    cat("\\nFirst 10 values from ORIGINAL RDS:\\n")
    print(head(col_data, 10))
    
    # Print unique values and their counts
    cat("\\nUnique values and counts from ORIGINAL RDS:\\n")
    print(table(col_data))
    
    # If it's a factor, show levels
    if (is.factor(col_data)) {{
        cat("\\nFactor levels:\\n")
        print(levels(col_data))
    }}
    
    cat("\\n=== COMPARISON CHECK ===\\n")
    cat("Are there 'Macrophage' entries?", any(grepl("Macrophage", col_data)), "\\n")
    cat("Are there 'Monocyte' entries?", any(grepl("Monocyte", col_data)), "\\n")
    
}} else {{
    cat("Column '{column_name}' not found!\\n")
    cat("Available columns:\\n")
    print(colnames(obj@meta.data))
}}
'''
    
    # Write the R script
    with open(r_script_path, 'w') as f:
        f.write(r_script_content)
    
    print(f"R script created at: {r_script_path}")
    print("\n=== RUN THIS R SCRIPT TO CHECK ORIGINAL DATA ===")
    print(f"Rscript {r_script_path}")
    print("\nThis will show you exactly what's in the original RDS file.")
    
    return None


# Alternative approaches for reading Seurat RDS files without rpy2  
def read_seurat_alternative(file_path):
    """
    Alternative methods to read Seurat RDS files:
    1. Check original RDS directly first
    2. Convert RDS to h5ad in R first
    3. Use existing import functions
    4. Manual conversion pipeline
    """
    
    # Method 0: Create R script to check original data
    print("=== STEP 1: CREATING R SCRIPT TO CHECK ORIGINAL RDS ===")
    check_rds_column_directly(file_path)
    
    # Method 1: Check if already converted to h5ad
    h5ad_path = file_path.replace('.rds', '.h5ad')
    try:
        print(f"Trying to read converted h5ad file: {h5ad_path}")
        adata = anndata.read_h5ad(h5ad_path)
        
        # Method 1a: Try to read metadata CSV to fix categorical conversion issues
        csv_path = file_path.replace('.rds', '_metadata.csv')
        try:
            print(f"Trying to read metadata CSV to fix categorical issues: {csv_path}")
            metadata_df = pd.read_csv(csv_path, index_col=0)
            
            # Replace the obs data with the properly formatted metadata
            print(f"Replacing obs data with CSV metadata")
            print(f"Original obs shape: {adata.obs.shape}")
            print(f"CSV metadata shape: {metadata_df.shape}")
            
            # Ensure indices match
            common_cells = adata.obs.index.intersection(metadata_df.index)
            print(f"Common cells between h5ad and CSV: {len(common_cells)}")
            
            if len(common_cells) > 0:
                # Replace obs with CSV data for common cells
                adata_subset = adata[common_cells].copy()
                adata_subset.obs = metadata_df.loc[common_cells]
                
                print(f"Successfully merged h5ad with CSV metadata")
                print(f"Final shape: {adata_subset.shape}")
                
                # Check if revised_annotations_10_5 now has proper values
                if "revised_annotations_10_5" in adata_subset.obs.columns:
                    annotations = adata_subset.obs["revised_annotations_10_5"]
                    print(f"Fixed annotations - first 5: {annotations.head().tolist()}")
                    print(f"Fixed annotations - unique count: {annotations.nunique()}")
                
                return adata_subset
            else:
                print("No common cells found between h5ad and CSV")
                
        except Exception as e:
            print(f"Could not read or merge CSV metadata: {e}")
            print("Proceeding with original h5ad file (with integer indices)")
        
        return adata
        
    except Exception as e:
        print(f"No h5ad file found: {e}")
    
    # Method 2: Check if CSV export exists (standalone)
    csv_path = file_path.replace('.rds', '_metadata.csv')
    try:
        print(f"Trying to read standalone CSV metadata: {csv_path}")
        metadata = pd.read_csv(csv_path, index_col=0)
        print(f"Found metadata with {len(metadata)} cells and {len(metadata.columns)} columns")
        
        # Show sample of the data
        if "revised_annotations_10_5" in metadata.columns:
            annotations = metadata["revised_annotations_10_5"]
            print(f"CSV annotations - first 5: {annotations.head().tolist()}")
            print(f"CSV annotations - unique: {annotations.unique()[:10]}")  # First 10 unique values
        
        return None  # Would need count matrix too for full AnnData
        
    except Exception as e:
        print(f"No CSV metadata found: {e}")
    
    print("Could not read Seurat file with available methods")
    print("\n=== SUGGESTED R SCRIPT TO EXPORT METADATA ===")
    print("Run this in R to export the metadata properly:")
    print("```r")
    print(f"library(Seurat)")
    print(f"obj <- readRDS('{file_path}')")
    print(f"metadata <- obj@meta.data")
    print(f"write.csv(metadata, '{csv_path}')")
    print("```")
    print("\nThis will export the metadata with proper categorical labels instead of integer indices.")
    
    return None


def makeFigure():
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)

    # Try to read the Seurat RDS file using alternative methods
    seurat_file = "/opt/BAL-scRNAseq.rds"
    
    print(f"Attempting to read Seurat file: {seurat_file}")
    X = read_seurat_alternative(seurat_file)
    
    if X is not None:
        print("\n=== DEBUGGING CONVERSION ISSUES ===")
        
        # Check if the column exists
        if "revised_annotations_10_5" in X.obs.columns:
            print(f"Column 'revised_annotations_10_5' found!")
            
            # Print basic info about the column
            annotations = X.obs["revised_annotations_10_5"]
            print(f"Data type: {type(annotations)}")
            print(f"Pandas dtype: {annotations.dtype}")
            print(f"Total cells: {len(annotations)}")
            print(f"Non-null values: {annotations.notna().sum()}")
            print(f"Null values: {annotations.isna().sum()}")
            
            # Print first 10 values (like R's head)
            print(f"\nFirst 10 values:")
            for i, (idx, val) in enumerate(annotations.head(10).items()):
                print(f"{idx}: {val}")
            
            # Print unique values and their counts
            print(f"\nUnique values and counts:")
            value_counts = annotations.value_counts(dropna=False)
            print(f"Number of unique values: {len(value_counts)}")
            for val, count in value_counts.head(20).items():  # Show top 20
                print(f"  '{val}': {count}")
            
            # Check if it's categorical
            if hasattr(annotations, 'cat'):
                print(f"\nCategorical info:")
                print(f"Categories: {annotations.cat.categories}")
                print(f"Number of categories: {len(annotations.cat.categories)}")
            
            # Compare with what you expect from R
            expected_values = ["Macrophage 1", "Monocyte", "Macrophage 8", "Macrophage 2"]
            print(f"\nChecking for expected values from R:")
            for val in expected_values:
                count = (annotations == val).sum()
                print(f"  '{val}': {count} occurrences")
                
        else:
            print(f"Column 'revised_annotations_10_5' NOT FOUND!")
            print(f"Available columns: {list(X.obs.columns)}")
            
            # Look for similar columns
            similar_cols = [col for col in X.obs.columns if 'annotation' in col.lower()]
            if similar_cols:
                print(f"Similar annotation columns found: {similar_cols}")
                
                # Check the first similar column
                first_col = similar_cols[0]
                print(f"\nExamining '{first_col}' instead:")
                alt_annotations = X.obs[first_col]
                print(f"First 10 values:")
                for i, (idx, val) in enumerate(alt_annotations.head(10).items()):
                    print(f"{idx}: {val}")
    else:
        print("Failed to load data - falling back to import_alad")
        X = import_alad(gene_threshold=0, normalize=True)
        
        print("\n=== CHECKING FALLBACK DATA ===")
        if "revised_annotations_10_5" in X.obs.columns:
            annotations = X.obs["revised_annotations_10_5"]
            print(f"Found in fallback data!")
            print(f"First 10 values:")
            for i, (idx, val) in enumerate(annotations.head(10).items()):
                print(f"{idx}: {val}")
            print(f"Unique values: {annotations.unique()}")
        else:
            print(f"'revised_annotations_10_5' not in fallback data either")
            print(f"Available columns: {list(X.obs.columns)}")
    
    return f
