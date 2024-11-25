import numpy as np
import tensorly as tl
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt
import autograd.numpy as anp


def project_tensor(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """

    B = np.zeros((proj_matrix.shape[1], proj_matrix.shape[1], tensor.shape[2]))
    for i in range(tensor.shape[2]):
        B[:, :, i] = proj_matrix.T @ tensor[:, :, i] @ proj_matrix

    return B


def project_data(
    X_list: list,
    factors: list[np.ndarray],
    weights: np.ndarray = None,
    full_tensor: np.ndarray = None,
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
    projected_X = np.empty((A.shape[0], B.shape[0], C.shape[0], D.shape[0]))

    if full_tensor is None:
        weights = np.ones(A.shape[1]) if weights is None else weights
        full_tensor = tl.cp_tensor.cp_to_tensor((weights, [A, B, C, D]))

    for i, mat in enumerate(X_list):
        lhs = full_tensor[i, :, :, :]
        cells = mat.shape[0]
        ces = lhs.shape[0]

        manifold = Stiefel(cells, ces)
        a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(lhs)

        @pymanopt.function.autograd(manifold)
        def objective_function(proj):
            a_mat_recon = anp.zeros_like(a_mat)
            for j in range(a_lhs.shape[2]):
                slice = anp.dot(anp.dot(proj, a_lhs[:, :, j]), proj.T)
                tensor = np.zeros((*slice.shape, a_lhs.shape[2]))
                # Create a mask of zeros with 1 at index j along last axis
                mask = np.zeros(a_lhs.shape[2])
                mask[j] = 1

                # Broadcast the mask and multiply with the expanded matrix
                a_mat_recon = anp.add(
                    a_mat_recon, np.expand_dims(slice, axis=-1) * mask
                )

            dif = a_mat - a_mat_recon
            return anp.sum(anp.abs(dif))

        problem = Problem(manifold=manifold, cost=objective_function)

        # Solve the problem
        solver = ConjugateGradient(verbosity=2)
        proj = solver.run(problem).point

        projections.append(proj)

        projected_X[i, :, :, :] = project_tensor(mat, proj)

    return projections, projected_X
