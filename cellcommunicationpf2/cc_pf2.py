import numpy as np
import tensorly as tl
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
import pymanopt
import autograd.numpy as anp


def project_data(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """
    return np.einsum("ab,cd,acg->bdg", proj_matrix, proj_matrix, tensor)


def solve_projections(
    X_list: list,
    full_tensor: np.ndarray,
) -> list[np.ndarray]:
    """
    Takes a list of 3D tensors of C x C x LR, a means matrix, factors of
    A: obs x rank
    B: CES x rank
    C: CES x rank
    D: LR x rank
    and solves for the projection matrices for each tensor as well as
    reconstruct the data based on the projection matrices.
    """
    projections: list[np.ndarray] = []

    for i, mat in enumerate(X_list):
        manifold = Stiefel(mat.shape[0], full_tensor.shape[1])
        a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(full_tensor[i, :, :, :])

        @pymanopt.function.autograd(manifold)
        def objective_function(proj):
            a_mat_recon = anp.einsum("ab,cd,acg->bdg", proj.T, proj.T, a_lhs)
            return anp.sum(anp.square(a_mat - a_mat_recon))

        problem = Problem(manifold=manifold, cost=objective_function)

        # Solve the problem
        solver = ConjugateGradient(verbosity=1)
        proj = solver.run(problem).point

        projections.append(proj)

    return projections
