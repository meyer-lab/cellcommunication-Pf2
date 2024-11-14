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

    # Reshape to 4D and average
    reshaped = matrix.reshape(b, b, X_dim, X_dim) # maybe redo this reshape manually
    return np.mean(reshaped, axis=(1, 3))


def project_tensor(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """
    proj_matrix = cp.asarray(proj_matrix)

    tensor = np.tensordot(tensor, proj_matrix, axes=([1], [0]))  # C × LR × CES
    tensor = np.transpose(tensor, (0, 2, 1))

    tensor = np.tensordot(proj_matrix.T, tensor, axes=([1], [0]))  # CES × CES × LR

    return tensor


def project_data(
    X_list: list, means: np.ndarray, factors: list[np.ndarray]
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
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0], D.shape[0])) # Having trouble understanding how to manipulate this for the 4th dimension
    means = cp.array(means)

    rank = A.shape[1]
    weights = np.ones(rank)
    full_tensor = tl.cp_tensor.cp_to_tensor((weights, [A, B, C, D]))

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat)

        lhs = full_tensor[i, :, :, :]
        # print(lhs.shape)
        lhs = lhs.reshape(lhs.shape[0] * lhs.shape[1], lhs.shape[2])
        lhs = lhs.T

        mat = cp.asarray(mat)
        lhs = cp.asarray(lhs)
        # print(mat.shape, lhs.shape, means.shape)
        U, _, Vh = cp.linalg.svd(mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2]) @ lhs, full_matrices=False)
        proj = U @ Vh
        proj = convert_4d_to_2d(
            cp.asnumpy(proj)
        )  # Perform the conversion here since we expect that
        projections.append(proj) 

        # Account for centering (currently not completed)
        # centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :, :] = project_tensor(mat, proj) #- centering # unflatten mat and then store projectedX with an extra dimension to store the full tensor

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

if __name__ == "__main__":
    unittest.main()
