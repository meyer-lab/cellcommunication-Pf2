import numpy as np
import pytest

from ..cc_pf2 import cc_pf2_redesigned
from ..import_data import (
    add_cond_idxs,
    anndata_lrp_overlap,
    import_balf_covid,
    import_ligand_receptor_pairs,
)


@pytest.mark.filterwarnings("ignore::anndata._warnings.OldFormatWarning")
@pytest.mark.parametrize("test_rank", [2, 3, 4])
@pytest.mark.parametrize("random_state", [42, 123, 456])
def test_cc_pf2_real_data(test_rank, random_state):
    """Test that cc_pf2_redesigned runs with different ranks and random states."""
    adata = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()

    # Filter data to include only genes in the ligand-receptor pairs
    adata_filtered, lr_pairs_filtered = anndata_lrp_overlap(adata, lr_pairs)

    # Get condition information and add indices
    condition_column = "condition"
    conditions = adata_filtered.obs[condition_column].unique()
    adata_filtered = add_cond_idxs(adata_filtered, condition_column)

    # Subset data for testing
    n_cells_per_condition = 100
    subset_cells = []
    for condition in conditions:
        mask = adata_filtered.obs[condition_column] == condition
        condition_cells = np.where(mask)[0]

        if len(condition_cells) > n_cells_per_condition:
            selected = np.random.RandomState(random_state).choice(
                condition_cells, n_cells_per_condition, replace=False
            )
            subset_cells.extend(selected)

    adata_subset = adata_filtered[subset_cells]
    rank = min(test_rank, len(conditions))
    n_iter_max = 10
    tol = 1e-2

    try:
        # Run cc_pf2_redesigned
        results, r2x = cc_pf2_redesigned(
            adata_subset, rank, n_iter_max, tol, random_state=random_state
        )
        factors, projections = results

        # Validate factors
        assert len(factors) == 4, f"Expected 4 factors, got {len(factors)}"

        # Expected shapes for factors
        assert factors[0].shape == (len(conditions), rank), f"Factor 0 (Conditions) shape mismatch: expected {(len(conditions), rank)}, got {factors[0].shape}"
        assert factors[1].shape == (rank, rank), f"Factor 1 (Sender cells) shape mismatch: expected {(rank, rank)}, got {factors[1].shape}"
        assert factors[2].shape == (rank, rank), f"Factor 2 (Receiver cells) shape mismatch: expected {(rank, rank)}, got {factors[2].shape}"
        assert factors[3].shape == (len(lr_pairs_filtered), rank), f"Factor 3 (LR pairs) shape mismatch: expected {(len(lr_pairs_filtered), rank)}, got {factors[3].shape}"
        # Check for NaN values in all factors
        for i, factor in enumerate(factors):
            assert not np.isnan(factor).any(), f"Factor {i} contains NaN values"

        # Validate projections
        assert len(projections) == len(conditions), f"Expected {len(conditions)} projections, got {len(projections)}"
        for i, proj in enumerate(projections):
            assert proj.shape[1] == rank, f"Projection {i} should have {rank} columns, got {proj.shape[1]}"

        # Validate R2X
        assert 0 <= r2x <= 1.0, f"R2X should be between 0 and 1, got {r2x}"

    except Exception as e:
        pytest.fail(f"Test failed with rank={rank}, random_state={random_state}: {e}")
