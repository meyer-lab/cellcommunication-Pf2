import numpy as np

from ..cc_pf2 import (
    project_data,
    solve_projections,
    init,
    reconstruction_error,
    fit_pf2,
)


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


def test_project_data_output_proj_matrix():
    """
    Tests that the project data method is actually able to solve for the correct optimal projection matrix.
    Asserts that the projection matrices solved are the same.
    """
    # Define dimensions
    num_tensors = 3
    cells = 20
    variables = 10
    obs = 20
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
    projections = [np.random.rand(cells, rank) for _ in range(obs)]

    # Call the reconstruction_error method
    error = reconstruction_error(factors, X_list, projections)

    assert error >= 0


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
    (factors, _), error = fit_pf2(X_list, rank, 10, 0.1)

    assert error >= 0
    assert factors[0].shape == (obs, rank)
    assert factors[1].shape == (rank, rank)
    assert factors[2].shape == (rank, rank)
    assert factors[3].shape == (LR, rank)
