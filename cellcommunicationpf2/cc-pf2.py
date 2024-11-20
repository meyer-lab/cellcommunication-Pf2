import anndata
import numpy as np
import cupy as cp
import tensorly as tl
import unittest

def convert_4d_to_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a b^2 x X^2 matrix to a b x X matrix.
    """
    b = int(np.sqrt(matrix.shape[0]))
    X_dim = int(np.sqrt(matrix.shape[1]))

    reshaped = matrix.reshape(b, b, X_dim, X_dim)  # maybe redo this reshape manually
    return np.mean(reshaped, axis=(1, 3))


def project_tensor(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """
    
    B = cp.zeros((proj_matrix.shape[1], proj_matrix.shape[1], tensor.shape[2]))
    for i in range(tensor.shape[2]):
        B[:,:,i] = proj_matrix.T @ tensor[:,:,i] @ proj_matrix
    
    return B


def project_data(
    X_list: list, means: np.ndarray, factors: list[np.ndarray], weights: np.ndarray = None, full_tensor: np.ndarray = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Takes a list of 3D tensors of C x C x LR, a means matrix, factors of
    A: obs x rank
    B: C x rank
    C: C x rank
    D: LR x rank
    and solves for the projection matrices for each tensor as well as
    reconstruct the data based on the projection matrices.
    """
    A, B, C, D = factors

    projections: list[np.ndarray] = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0], D.shape[0]))
    means = cp.array(means)

    if full_tensor is None:    
        weights = np.ones(A.shape[1]) if weights is None else weights
        full_tensor = tl.cp_tensor.cp_to_tensor((weights, [A, B, C, D]))

    for i, mat in enumerate(X_list):
        lhs = full_tensor[i, :, :, :]
        mat = cp.asarray(mat)
        lhs = cp.asarray(lhs)
        
        flatenned_mat = mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2])
        flattened_lhs = lhs.reshape(lhs.shape[0] * lhs.shape[1],  lhs.shape[2])

        U, _, Vh = cp.linalg.svd(flatenned_mat @ flattened_lhs.T, full_matrices=False)
        proj = U @ Vh
        proj = convert_4d_to_2d(
            cp.asnumpy(proj)
        )  # Perform the conversion here since we expect that
        projections.append(proj) 

        # Account for centering (currently not completed)
        # centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :, :] = project_tensor(mat, cp.asarray(proj)) #- centering # unflatten mat and then store projectedX with an extra dimension to store the full tensor

    return projections, cp.asnumpy(projected_X)

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
        projections = [np.linalg.qr(np.random.rand(cells, rank))[0] for _ in range(num_tensors)]

        # Recreate the original tensor using the projection matrices and projected tensor
        recreated_tensors = []
        for i in range(num_tensors):
            Q = cp.asarray(projections[i])
            A = cp.asarray(projected_X[i, :, :, :])
            B = project_tensor(A, Q.T)
            recreated_tensors.append(B)

        # Call the project_data method using the recreated tensors to get the projected_X that gets solved by our method
        projections_recreated, _ = project_data(recreated_tensors, np.zeros(cells), [np.zeros((obs, rank)), np.zeros((rank, rank)), np.zeros((rank, rank)), np.zeros((LR, rank))], full_tensor=projected_X)

        # Assert that the projections are the same
        for i in range(num_tensors):
            difference_sum = np.sum(np.abs(projections[i] - projections_recreated[i]))
            print(f"Projection {i} difference sum: {difference_sum}. Sum of projections in absolute: {np.sum(np.abs(projections[i]))}. Sum of projections_recreated in absolute: {np.sum(np.abs(projections_recreated[i]))}")
            self.assertTrue(np.allclose(projections[i], projections_recreated[i]))


if __name__ == "__main__":
    unittest.main()
