import numpy as np
import sparse
from tensorly import cp_to_tensor
from tensorly.cp_tensor import cp_permute_factors, CPTensor
from ..cc_pf2 import (
    init,
    cc_pf2,
    project_data,
    solve_projections,
    init,
    reconstruction_error,
)


def dense_to_sparse(tensor, sparsity=0.9):
    """Convert dense tensor to sparse by randomly zeroing elements."""
    mask = np.random.random(tensor.shape) > sparsity
    sparse_data = tensor * mask
    return sparse.COO.from_numpy(sparse_data)

def test_init():
    """Tests initialization with sparse tensors."""
    obs = 3
    cells = 20
    LR = 10
    rank = 5

    # Generate sparse X_list
    X_list = [dense_to_sparse(np.random.rand(cells, cells, LR)) for _ in range(obs)]

    factors = init(X_list, rank)
    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)

def test_project_data():
    """Tests projection with sparse tensor."""
    cells = 20
    LR = 10
    rank = 5

    # Generate sparse tensor
    X_mat = dense_to_sparse(np.random.rand(cells, cells, LR))
    proj_matrix = np.linalg.qr(np.random.rand(cells, rank))[0]
    
    projected_X = project_data(X_mat, proj_matrix)
    assert projected_X.shape == (rank, rank, LR)

def test_project_data_output_proj_matrix():
    """Tests optimal projection matrix calculation with sparse data."""
    cells = 20
    LR = 10
    rank = 5
    obs = 3

    # Generate reference tensors and projections
    projected_X = dense_to_sparse(np.random.rand(obs, rank, rank, LR))
    projections = [np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(obs)]

    recreated_tensors = []
    for i in range(obs):
        Q = projections[i]
        A = projected_X[i, :, :, :]
        B = project_data(A, Q.T)
        recreated_tensors.append(B)

    # Keep your original function call order
    projections_recreated = solve_projections(recreated_tensors, projected_X.todense())

    # Verify projections match (up to sign)
    for i in range(obs):
        sign_correct = np.sign(projections[i][0, 0] * projections_recreated[i][0, 0])
        np.testing.assert_allclose(
            projections[i], 
            projections_recreated[i] * sign_correct, 
            atol=1e-2
        )

def test_reconstruction_error():
    """Tests reconstruction error with sparse tensors."""
    cells = 20
    LR = 10
    rank = 5
    obs = 3

    # Generate sparse X_list
    X_list = [dense_to_sparse(np.random.rand(cells, cells, LR)) for _ in range(obs)]
    
    factors = [
        np.random.rand(obs, rank),
        np.random.rand(rank, rank),
        np.random.rand(rank, rank),
        np.random.rand(LR, rank),
    ]
    
    projections = [
        np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(obs)
    ]

    error = reconstruction_error(factors, X_list, projections)
    assert error >= 0

def test_fitting_method():
    """Tests fitting with sparse tensors."""
    cells = 20
    LR = 10
    rank = 5
    obs = 3

    X_list = [dense_to_sparse(np.random.rand(cells, cells, LR)) for _ in range(obs)]
    (factors, _), error = cc_pf2(X_list, rank, 2, 0.1)

    assert error >= 0
    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)

def random_4d_tensor(obs, rank):
    """Generates random sparse 4D tensor."""
    shapes = []
    for _ in range(obs):
        cells = np.random.randint(10, 20)
        LR = np.random.randint(10, 20)
        shapes.append((cells, cells, LR))

    projections = [
        np.linalg.qr(np.random.rand(shape[0], rank))[0] for shape in shapes
    ]

    factors = [
        np.random.rand(obs, rank),
        np.random.rand(rank, rank),
        np.random.rand(rank, rank),
        np.random.rand(LR, rank),
    ]

    reconstructed_X = cp_to_tensor((None, factors))
    X_list = [dense_to_sparse(reconstructed_X[i]) for i in range(obs)]

    return X_list, factors, projections

def test_fitting_method_output_reproducible():
    """Tests output reproducibility with sparse tensors."""
    X_list, _, _ = random_4d_tensor(3, 5)

    (factors1, _), _ = cc_pf2(X_list, 5, 10, 1e-2, random_state=0)
    (factors2, _), _ = cc_pf2(X_list, 5, 10, 1e-2, random_state=0)

    cp1 = CPTensor((None, factors1))
    cp2 = CPTensor((None, factors2))
    cp2_permuted, _ = cp_permute_factors(cp1, cp2)

    for i, (f1, f2) in enumerate(zip(cp1.factors, cp2_permuted.factors)):
        assert np.allclose(f1, f2, rtol=1e-2, atol=1e-2)
