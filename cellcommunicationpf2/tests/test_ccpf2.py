import numpy as np

from ..cc_pf2 import (
    project_data,
    solve_projections,
    init,
    reconstruction_error,
    fit_pf2,
)

from tensorly import cp_to_tensor
from tensorly.cp_tensor import cp_permute_factors, CPTensor

import scipy.sparse as sp
import sparse

import pytest


def dense_to_sparse(tensor, sparsity=0.9):
    """Convert dense tensor to sparse by randomly zeroing elements."""
    mask = np.random.random(tensor.shape) > sparsity
    sparse_data = tensor * mask
    return sparse.COO.from_numpy(sparse_data)


@pytest.mark.skip(reason="This test is for dense data")
def test_init():
    """
    Tests that the dimensions are correct and that the method is able to run without errors.
    """

    # Define dimensions
    obs = 3
    cells = 20
    LR = 10
    rank = 5

    # Generate random X_list
    X_list = [np.random.rand(cells, cells, LR) for _ in range(obs)]

    # Call the init method
    factors = init(X_list, rank)

    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)


@pytest.mark.skip(reason="This test is for dense data")
def test_project_data():
    """
    Tests that the dimensions are correct and that the method is able to run without errors.
    """

    # Define dimensions
    cells = 20
    LR = 10
    rank = 5

    # Generate random X_list
    X_mat = np.random.rand(cells, cells, LR)

    # Projection matrix
    proj_matrix = np.linalg.qr(np.random.rand(cells, rank))[0]

    projected_X = project_data(X_mat, proj_matrix)

    assert projected_X.shape == (rank, rank, LR)


def test_project_data_sparse():
    """Tests projection with sparse tensor."""
    cells = 20
    LR = 10
    rank = 5

    # Generate sparse tensor
    X_mat = dense_to_sparse(np.random.rand(cells, cells, LR))
    proj_matrix = np.linalg.qr(np.random.rand(cells, rank))[0]

    projected_X = project_data(X_mat, proj_matrix)
    assert projected_X.shape == (rank, rank, LR)


@pytest.mark.skip(reason="This test is for dense data")
def test_project_data_output_proj_matrix():
    """
    Tests that the project data method is actually able to solve for the correct optimal projection matrix.
    Asserts that the projection matrices solved are the same.
    """
    # Define dimensions
    num_tensors = 3
    cells = 20
    variables = 10
    obs = num_tensors
    rank = 5
    # Generate a random projected tensor
    projected_X = np.random.rand(obs, rank, rank, variables)

    # Generate a random set of projection matrices
    projections = [
        np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(num_tensors)
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
        recreated_tensors,
        projected_X,
    )

    # Assert that the projections are the same
    for i in range(num_tensors):
        sign_correct = np.sign(projections[i][0, 0] * projections_recreated[i][0, 0])
        np.testing.assert_allclose(
            projections[i], projections_recreated[i] * sign_correct, atol=1e-9
        )


def test_project_data_sparse_input():
    """
    Tests that the solve_projections method correctly handles sparse input tensors.
    Creates sparse tensors, solves for projection matrices, and compares results
    with the same operation on equivalent dense tensors.
    """
    # Define dimensions
    num_tensors = 3
    cells = 20
    variables = 10
    obs = num_tensors
    rank = 5

    # Generate a random projected tensor
    projected_X = dense_to_sparse(np.random.rand(obs, rank, rank, variables))

    # Generate a random set of projection matrices
    projections = [
        np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(num_tensors)
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
        recreated_tensors,
        projected_X,
    )

    # Assert that the projections are the same
    for i in range(num_tensors):
        # Calculate sign correction factor
        sign_correct = np.sign(projections[i][0, 0] * projections_recreated[i][0, 0])
        np.testing.assert_allclose(
            projections[i], projections_recreated[i] * sign_correct, atol=1e-9
        )


@pytest.mark.skip(reason="This test is for dense data")
def test_reconstruction_error():
    """
    Tests that the reconstruction error function is able to run without errors. ie. the dimensions are correct.
    """

    # Define dimensions
    cells = 20
    LR = 10
    rank = 5
    obs = 3

    # Generate random X_list
    X_list = [np.random.rand(cells, cells, LR) for _ in range(obs)]

    # Generate random factors
    factors = [
        np.random.rand(obs, rank),
        np.random.rand(rank, rank),
        np.random.rand(rank, rank),
        np.random.rand(LR, rank),
    ]

    # Generate random projections
    projections = projections = [
        np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(obs)
    ]

    # Call the reconstruction_error method
    error = reconstruction_error(factors, X_list, projections)

    assert error >= 0


@pytest.mark.skip(reason="This test is for dense data")
def test_fitting_method():
    """
    Tests the fitting method to ensure that it is able to run without errors ie. the dimensions are correct.
    """

    # Define dimensions
    cells = 20
    LR = 10
    rank = 5
    obs = 3

    # Generate random X_list
    X_list = [np.random.rand(cells, cells, LR) for _ in range(obs)]

    # Call the fitting method
    (factors, _), error = fit_pf2(X_list, rank, 2, 0.1)

    assert error >= 0
    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)


@pytest.mark.skip(reason="This test is for dense data")
def test_fitting_method_output_reproducible():
    """
    Tests that the output of the decomposition is the same between two runs of the fitting method.
    """

    X_list, _, _ = random_4d_tensor(3, 5)

    (factors1, _), _ = fit_pf2(X_list, 5, 10, 1e-2, random_state=0)
    (factors2, _), _ = fit_pf2(X_list, 5, 10, 1e-2, random_state=0)

    cp1 = CPTensor((None, factors1))
    cp2 = CPTensor((None, factors2))

    cp2_permuted, _ = cp_permute_factors(cp1, cp2)

    f1s = cp1.factors
    f2s = cp2_permuted.factors

    for i, (f1, f2) in enumerate(zip(f1s, f2s)):
        max_diff = np.max(np.abs(f1 - f2))
        print(f"Max difference in factor {i}: {max_diff}")
        assert np.allclose(f1, f2, rtol=1e-2, atol=1e-2)


def random_4d_tensor(obs, rank):
    """
    Generates a random 4D tensor with the given number of observations and rank.
    Generated tensor will be of the form obs x cells x cells x LR
    """

    # Generate a list of random dimensions for the tensors
    shapes = []
    for _ in range(obs):
        cells = np.random.randint(10, 20)
        LR = np.random.randint(10, 20)
        shapes.append((cells, cells, LR))

    projections = [np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(obs)]

    # Generate random factors
    factors = [
        np.random.rand(obs, rank),
        np.random.rand(rank, rank),
        np.random.rand(rank, rank),
        np.random.rand(LR, rank),
    ]

    # Generate X_list from the factors and projections
    reconstructed_X = cp_to_tensor((None, factors))
    X_list = [reconstructed_X[i, :, :, :] for i in range(obs)]

    return X_list, factors, projections
