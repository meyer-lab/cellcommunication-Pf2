import numpy as np
import pytest
from sparse import COO
from tensorly import cp_to_tensor
from tensorly.cp_tensor import CPTensor, cp_permute_factors

from ..cc_pf2 import cc_pf2, init, project_data, reconstruction_error, solve_projections


def dense_to_sparse(tensor):
    """Convert a dense tensor to sparse by randomly zeroing elements with uniform sampling."""
    # rng = np.random.default_rng(random_state)
    # mask = rng.uniform(0.0, 1.0, size=tensor.shape) > sparsity
    # sparse_data = tensor * mask
    return COO.from_numpy(tensor)


def random_4d_tensor(
    obs: int, rank: int, cell_sizes: list[int] = None, LR: int = None, random_state=None
):
    """Generate a list of random dense 4D tensors using uniform sampling."""
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
        rng.uniform(0.0, 1.0, size=(rank, rank)),
        rng.uniform(0.0, 1.0, size=(LR, rank)),
    ]
    reconstructed = cp_to_tensor((None, factors))
    X_list = [
        project_data(reconstructed[i], proj.T) for i, proj in enumerate(projections)
    ]
    return X_list, factors, projections


def random_4d_tensor_sparse(
    obs: int, rank: int, cell_sizes: list[int] = None, LR: int = None, random_state=None
):
    """Generate a list of random sparse 4D tensors using uniform sampling."""
    # Fixed function call: use named parameters to avoid ordering issues
    X_list, factors, projections = random_4d_tensor(
        obs=obs, rank=rank, cell_sizes=cell_sizes, LR=LR, random_state=random_state
    )
    sparse_list = [dense_to_sparse(X) for X in X_list]
    return sparse_list, factors, projections


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [3, 4, 5, 6])
def test_init(sparse, random_state):
    obs, rank, LR = 3, 5, 10

    if sparse:
        X_list, _, _ = random_4d_tensor_sparse(
            obs, rank, LR=LR, random_state=random_state
        )
    else:
        X_list, _, _ = random_4d_tensor(obs, rank, LR=LR, random_state=random_state)

    factors = init(X_list, rank, random_state=random_state)

    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [3, 4, 5, 6])
def test_project_data(sparse, random_state):
    rng = np.random.default_rng(random_state)
    cells, LR, rank = 20, 10, 5

    dense = rng.uniform(0.0, 1.0, size=(cells, cells, LR))
    X_mat = dense_to_sparse(dense) if sparse else dense

    proj_matrix = np.linalg.qr(rng.uniform(0.0, 1.0, size=(cells, rank)))[0]
    projected_X = project_data(X_mat, proj_matrix)

    assert projected_X.shape == (rank, rank, LR)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [3, 4, 5, 6, 7, 8, 9, 10])
def test_project_data_output_proj_matrix(sparse, random_state):
    rng = np.random.default_rng(random_state)
    num_tensors, cells, variables, rank = 2, 20, 50, 5

    projected = rng.uniform(0.0, 1.0, size=(num_tensors, rank, rank, variables))
    projected_X = dense_to_sparse(projected) if sparse else projected
    projections = [
        np.linalg.qr(rng.uniform(0.0, 1.0, size=(cells, rank)))[0]
        for _ in range(num_tensors)
    ]

    # Recreate the original tensor using the projection matrices and projected tensor
    recreated_tensors = []
    for i in range(num_tensors):
        Q = projections[i]
        A = projected_X[i, :, :, :]
        B = project_data(A, Q.T)
        recreated_tensors.append(B)

    # Call the project_data method using the recreated tensors to get the projected_X that gets solved by our method
    projections_recreated = solve_projections(
        recreated_tensors, projected_X, random_state
    )

    # Assert that the projections are the same
    for i in range(num_tensors):
        sign_correct = np.sign(projections[i][0, 0] * projections_recreated[i][0, 0])
        np.testing.assert_allclose(
            projections[i],
            projections_recreated[i] * sign_correct,
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [3, 4, 5, 6])
def test_reconstruction_error(sparse, random_state):
    obs, rank = 3, 5

    if sparse:
        X_list, factors, projections = random_4d_tensor_sparse(
            obs, rank, random_state=random_state
        )
    else:
        X_list, factors, projections = random_4d_tensor(
            obs, rank, random_state=random_state
        )

    error = reconstruction_error(factors, X_list, projections)

    assert error >= 0


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [3, 4, 5, 6])
def test_fitting_method(sparse, random_state):
    obs, rank, LR = 3, 5, 10

    if sparse:
        X_list, _, _ = random_4d_tensor_sparse(
            obs, rank, LR=LR, random_state=random_state
        )
    else:
        X_list, _, _ = random_4d_tensor(obs, rank, LR=LR, random_state=random_state)

    (facs, _), error = cc_pf2(X_list, rank, 2, 0.1, random_state=random_state)

    assert error >= 0
    assert facs[0].shape == (obs, rank)
    assert facs[1].shape == (rank, rank)
    assert facs[2].shape == (rank, rank)
    assert facs[3].shape == (LR, rank)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("random_state", [0, 1, 2])
def test_fitting_method_output_reproducible(sparse, random_state):
    obs, rank = 3, 5

    if sparse:
        X_list, _, _ = random_4d_tensor_sparse(obs, rank, random_state=random_state)
    else:
        X_list, _, _ = random_4d_tensor(obs, rank, random_state=random_state)

    (f1, _), _ = cc_pf2(X_list, rank, 10, 1e-2, random_state=random_state)
    (f2, _), _ = cc_pf2(X_list, rank, 10, 1e-2, random_state=random_state)

    cp1 = CPTensor((None, f1))
    cp2 = CPTensor((None, f2))

    cp2p, _ = cp_permute_factors(cp1, cp2)

    for f1, f2 in zip(cp1.factors, cp2p.factors, strict=False):
        assert np.allclose(f1, f2, rtol=1e-2, atol=1e-2)
