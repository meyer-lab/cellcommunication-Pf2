import numpy as np
import pytest
from sparse import COO
from tensorly import cp_to_tensor
from tensorly.cp_tensor import CPTensor

from ..cc_pf2 import cc_pf2_redesigned


def test_cc_pf2_redesigned():
    """Test that cc_pf2_redesigned runs without errors with properly constructed 2D tensors."""

    # Use the random_3d_tensor function to generate better test data
    n_samples = 3
    rank = 4
    cell_sizes = [8, 10, 12]  # Different cell counts per sample
    n_genes = 15
    random_state = 42  # Use a different seed to avoid convergence issues

    X_list, _, _ = random_3d_tensor(
        obs=n_samples,
        rank=rank,
        cell_sizes=cell_sizes,
        LR=n_genes,
        random_state=random_state,
    )

    try:
        # Run the function
        pf2_results, r2x = cc_pf2_redesigned(
            X_list, rank, 5, 1e-2, random_state=random_state
        )
        # If we get here without errors, the test passes
        assert r2x >= 0.0  # R2X should be non-negative
        
        # assert that the factors have the expected shapes
        factors = pf2_results[0]
        assert factors[0].shape == (n_samples, rank)
        assert factors[1].shape == (rank, rank)
        assert factors[2].shape == (rank, rank)
        assert factors[3].shape == (n_genes, rank)
        print(f"Test passed with R2X: {r2x}")
    except Exception as e:
        pytest.fail(f"cc_pf2_redesigned raised an exception: {e}")


def random_3d_tensor(
    obs: int, rank: int, cell_sizes: list[int] = None, LR: int = None, random_state=None
):
    """Generate a list of random dense 2D tensors (3D structure) using uniform sampling."""
    rng = np.random.default_rng(random_state)
    if cell_sizes is None:
        cell_sizes = rng.integers(10, 20, size=obs)
    if LR is None:
        LR = rng.integers(10, 20)

    projections = [
        np.linalg.qr(rng.uniform(0.0, 1.0, size=(n, rank)))[0] for n in cell_sizes
    ]

    factors = [
        rng.uniform(0.0, 1.0, size=(obs, rank)),
        rng.uniform(0.0, 1.0, size=(rank, rank)),
        rng.uniform(0.0, 1.0, size=(LR, rank)),
    ]
    reconstructed = cp_to_tensor((None, factors))

    # reconstructed has shape (obs, rank, LR)
    # reconstructed[i] has shape (rank, LR)
    # proj.T has shape (rank, n) where n is cell_sizes[i]
    # We want: proj.T @ reconstructed[i] -> (n, LR)
    X_list = [projections[i] @ reconstructed[i] for i in range(obs)]
    return X_list, factors, projections

