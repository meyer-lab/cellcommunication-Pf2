import unittest


class TestProjectData(unittest.TestCase):
    def test_project_data(self):
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
        self.assertEqual(len(projections), num_tensors)
        for proj in projections:
            self.assertEqual(proj.shape, (cells, rank))

        self.assertEqual(projected_X.shape, (obs, rank, rank, LR))

        # Check that projected_X contains finite numbers
        self.assertTrue(np.all(np.isfinite(projected_X)))

    # Test the output of the project_data method to be correct
    def test_project_data_output(self):
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
            self.assertTrue(np.allclose(projections[i], projections_recreated[i]))


if __name__ == "__main__":
    unittest.main()
