import anndata
import numpy as np
import cupy as cp
import tensorly as tl
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions, ConjugateGradient
import pymanopt

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
    B: CES x rank
    C: CES x rank
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
        cells = mat.shape[0]
        ces = lhs.shape[0]

        manifold = Stiefel(cells, ces)

        @pymanopt.function.numpy(manifold)
        def objective_function(proj):
            return np.linalg.norm(mat - project_tensor(lhs, cp.asarray(proj.T)), 'fro')

        problem = Problem(manifold=manifold, cost=objective_function)

        # Solve the problem
        solver = TrustRegions()
        proj = solver.run(problem).point

        # flatenned_mat = mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2])
        # flattened_lhs = lhs.reshape(lhs.shape[0] * lhs.shape[1],  lhs.shape[2])

        # U, _, Vh = cp.linalg.svd(flatenned_mat @ flattened_lhs.T, full_matrices=False)
        # proj = U @ Vh
        # proj = convert_4d_to_2d(
        #     cp.asnumpy(proj)
        # )  # Perform the conversion here since we expect that
        projections.append(proj) 

        # Account for centering (currently not completed)
        # centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :, :] = project_tensor(mat, cp.asarray(proj)) #- centering # unflatten mat and then store projectedX with an extra dimension to store the full tensor

    return projections, cp.asnumpy(projected_X)
