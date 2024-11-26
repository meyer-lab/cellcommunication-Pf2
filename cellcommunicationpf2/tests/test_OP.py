import numpy as np
from ..cc_pf2 import project_data, solve_projections
import pytest


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

    # Call the project_data method
    print(proj_matrix.shape)
    projected_X = project_data(X_mat, proj_matrix)

    assert projected_X.shape == (rank, rank, LR)


@pytest.mark.xfail(reason="The project method hasn't been completed yet")
def test_project_data_output():
    """
    Tests that the project data method is actually able to solve for the correct optimal projection matrix.
    """
    # Define dimensions
    num_tensors = 3
    cells = 20
    LR = 10
    obs = 5
    rank = 5
    # Generate a random projected tensor
    projected_X = np.random.rand(obs, rank, rank, LR)

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
        difference_sum = np.sum(np.abs(projections[i] - projections_recreated[i]))
        print(
            f"Projection {i} difference sum: {difference_sum}. Sum of projections in absolute: {np.sum(np.abs(projections[i]))}"
        )
        assert np.allclose(projections[i], projections_recreated[i])
