import anndata
import pandas as pd
import pytest

from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)


@pytest.mark.skip(reason="Test takes too long to run")
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
    assert X.shape[0] == X_original_shape[0], (
        "AnnData dimensions should not change after overlap"
    )
    assert len(df_lrp) <= df_lrp_original_len, (
        "Filtered LR pairs should be equal or fewer than original"
    )
    assert X.shape[1] <= X_original_shape[1], (
        "Filtered LR pairs should be equal or fewer than original"
    )

    # Test subsampling
    X_small = X[::200]
    number_of_pairs = 20
    df_lrp_small = df_lrp.iloc[:number_of_pairs, :]
