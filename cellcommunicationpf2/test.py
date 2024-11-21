import numpy as np
import cupy as cp
from cc_pf2 import project_data, project_tensor
import pytest


class TestProjectData:
    @pytest.mark.skip(reason="The project method hasn't been completed yet")
    def test_project_data(self):
        """
        Tests that the dimensions are correct and that the method is able to run without errors.
        """

        # Define dimensions
        num_tensors = 3
        cells = 400
        LR = 50
        obs = 10
        rank = 10

        # Generate random X_list
        X_list = [np.random.rand(cells, cells, LR) for _ in range(num_tensors)]

        # Generate random means matrix
        means = np.random.rand(cells)

        # Generate random factors: A (obs x rank), B (C x rank), C (C x rank), D (LR x rank)
        A = np.random.rand(obs, rank)
        B = np.random.rand(rank, rank)
        C = np.random.rand(rank, rank)
        D = np.random.rand(LR, rank)
        factors = [A, B, C, D]

        # Call the project_data method
        projections, projected_X = project_data(X_list, means, factors)

        # Assertions
        assert len(projections) == num_tensors
        for proj in projections:
            assert proj.shape == (cells, rank)

        assert projected_X.shape == (obs, rank, rank, LR)

    @pytest.mark.skip(reason="The project method hasn't been completed yet")
    def test_project_data_output(self):
        """
        Tests that the project data method is actually able to solve for the correct optimal projection matrix.
        """
        # Define dimensions
        num_tensors = 3
        cells = 400
        LR = 50
        obs = 10
        rank = 10
        # Generate a random projected tensor
        projected_X = np.random.rand(obs, rank, rank, LR)

        # Generate a random set of projection matrices
        projections = [
            np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(num_tensors)
        ]

        # Recreate the original tensor using the projection matrices and projected tensor
        recreated_tensors = []
        for i in range(num_tensors):
            Q = cp.asarray(projections[i])
            A = cp.asarray(projected_X[i, :, :, :])
            B = project_tensor(A, Q.T)
            recreated_tensors.append(B)

        # Call the project_data method using the recreated tensors to get the projected_X that gets solved by our method
        projections_recreated, _ = project_data(
            recreated_tensors,
            np.zeros(cells),
            [
                np.zeros((obs, rank)),
                np.zeros((rank, rank)),
                np.zeros((rank, rank)),
                np.zeros((LR, rank)),
            ],
            full_tensor=projected_X,
        )

        # Assert that the projections are the same
        for i in range(num_tensors):
            difference_sum = np.sum(np.abs(projections[i] - projections_recreated[i]))
            print(
                f"Projection {i} difference sum: {difference_sum}. Sum of projections in absolute: {np.sum(np.abs(projections[i]))}. Sum of projections_recreated in absolute: {np.sum(np.abs(projections_recreated[i]))}"
            )
            assert np.allclose(projections[i], projections_recreated[i])
