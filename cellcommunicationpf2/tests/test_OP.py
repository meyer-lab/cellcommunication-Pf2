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


def test_project_data_output_proj_data():
    """
    Tests that the project data method is actually able to solve for the correct optimal projection matrix.
    Asserts that the projected data through the solved matrices is the same as the input projectedX.
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

    # Assert that the projected tensors are the same
    for i in range(num_tensors):
        assert np.allclose(project_data(recreated_tensors[i], projections_recreated[i]), projected_X[i])


def test_project_data_output_proj_matrix():
    """
    Tests that the project data method is actually able to solve for the correct optimal projection matrix.
    Asserts that the projection matrices solved are the same.
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

    for i in range(len(projections)):
        proj = projections[i]  
        U, _, Vt = np.linalg.svd(proj, full_matrices=False)
        projections[i] = U @ Vt

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
        assert np.allclose(projections[i], projections_recreated[i]) or np.allclose(projections[i], -projections_recreated[i])

