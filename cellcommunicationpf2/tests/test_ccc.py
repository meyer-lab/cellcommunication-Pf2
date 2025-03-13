import anndata
import pandas as pd
from ..import_data import (
    import_balf_covid,
    import_ligand_receptor_pairs,
    anndata_lrp_overlap,
)
from ..ccc import calc_communication_score


def test_anndata_ccc_processing_pipeline():
    """Test the full data processing pipeline"""
    X = import_balf_covid()
    df_lrp = import_ligand_receptor_pairs()
    
    # Test that input objects have expected types
    assert isinstance(X, anndata.AnnData), "Import should return an AnnData object"
    assert isinstance(df_lrp, pd.DataFrame), "LR pairs should be in a DataFrame"
    
    # Test overlap function
    X_original_shape = X.shape
    df_lrp_original_len = len(df_lrp)
    
    X, df_lrp = anndata_lrp_overlap(X, df_lrp)
    
    # Check that anndata_lrp_overlap doesn't change AnnData dimensions
    assert X.shape[0] == X_original_shape[0], "AnnData dimensions should not change after overlap"
    assert len(df_lrp) <= df_lrp_original_len, "Filtered LR pairs should be equal or fewer than original"
    assert X.shape[1] <= X_original_shape[1], "Filtered LR pairs should be equal or fewer than original"
    
    # Test subsampling
    X_small = X[::200]
    number_of_pairs = 20
    df_lrp_small = df_lrp.iloc[:number_of_pairs, :]
    
    ccc_X = calc_communication_score(X_small, df_lrp_small, communication_score="expression_product")
        
    assert ccc_X.shape[1] == number_of_pairs, "Output should have same number of LR pairs as input"
    assert len(pd.unique(ccc_X.obs["sample"])) == len(pd.unique(X_small.obs["sample"])), "Output should have same number of samples as input"
  