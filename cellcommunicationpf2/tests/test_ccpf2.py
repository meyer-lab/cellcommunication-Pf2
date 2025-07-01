import numpy as np
import pytest
from tensorly import cp_to_tensor

from ..cc_pf2 import cc_pf2_redesigned
from ..import_data import import_balf_covid, import_ligand_receptor_pairs, anndata_lrp_overlap, add_cond_idxs

def random_3d_tensor(
    obs: int, rank: int, cell_sizes: list[int] = None, genes: int = None, random_state=None
):
    """Generate a list of random dense 2D tensors (3D structure) using uniform sampling."""
    rng = np.random.default_rng(random_state)
    if cell_sizes is None:
        cell_sizes = rng.integers(10, 20, size=obs)
    if genes is None:
        genes = rng.integers(10, 20)

    projections = [
        np.linalg.qr(rng.uniform(0.0, 1.0, size=(n, rank)))[0] for n in cell_sizes
    ]

    factors = [
        rng.uniform(0.0, 1.0, size=(obs, rank)),
        rng.uniform(0.0, 1.0, size=(rank, rank)),
        rng.uniform(0.0, 1.0, size=(genes, rank)),
    ]
    reconstructed = cp_to_tensor((None, factors))

    # reconstructed has shape (obs, rank, LR)
    # reconstructed[i] has shape (rank, LR)
    # proj.T has shape (rank, n) where n is cell_sizes[i]
    # We want: proj.T @ reconstructed[i] -> (n, LR)
    X_list = [projections[i] @ reconstructed[i] for i in range(obs)]
    return X_list, factors, projections


def test_cc_pf2_real_data():
    """Test that cc_pf2_redesigned runs without errors with real data."""

    adata = import_balf_covid()

    # Import ligand-receptor pairs
    lr_pairs = import_ligand_receptor_pairs()

    # Filter data to include only genes in the ligand-receptor pairs
    adata_filtered, _ = anndata_lrp_overlap(adata, lr_pairs)

    # From the test output, we know 'condition' is the column with 3 values:
    # 'Control', 'Moderate COVID-19', 'Severe COVID-19'
    condition_column = 'condition'
    print(f"Using '{condition_column}' as condition with values: {adata_filtered.obs[condition_column].unique()}")

    # Add condition indices using the determined condition column
    adata_filtered = add_cond_idxs(adata_filtered, condition_column)

    # Subset data for testing - use more cells for better stability
    n_cells_per_condition = 100  # Increased from 50
    conditions = adata_filtered.obs[condition_column].unique()

    subset_cells = []
    for condition in conditions:
        mask = adata_filtered.obs[condition_column] == condition
        condition_cells = np.where(mask)[0]

        if len(condition_cells) > n_cells_per_condition:
            # Use consistent random seed
            selected = np.random.RandomState(42).choice(
                condition_cells, n_cells_per_condition, replace=False
            )
            subset_cells.extend(selected)

    # Subset the data
    adata_subset = adata_filtered[subset_cells]

    # Parameters for decomposition - use lower rank for stability
    rank = min(2, len(conditions))  # Reduced from 3 to 2
    n_iter_max = 3  # Increased iterations slightly
    tol = 1e-2

    try:
        # Run cc_pf2_redesigned
        results, r2x = cc_pf2_redesigned(
            adata_subset, rank, n_iter_max, tol, random_state=42
        )

        # Basic validation
        factors, projections = results

        # Check that the factors have the expected dimensions
        assert factors[0].shape[0] == len(conditions)
        assert factors[0].shape[1] == rank

        # Check that we have one projection matrix per condition
        assert len(projections) == len(conditions)

        # Verify R2X value
        assert 0 <= r2x <= 1.0, f"R2X should be between 0 and 1, got {r2x}"

        print(f"Cell communication test with real data passed. R2X: {r2x:.4f}")

    except Exception as e:
        pytest.fail(f"cc_pf2_redesigned with real data raised an exception: {e}")
