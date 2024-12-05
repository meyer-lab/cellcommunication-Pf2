import autograd.numpy as anp
import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions


def project_data(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """
    return np.einsum("ab,cd,acg->bdg", proj_matrix, proj_matrix, tensor)


def solve_projections(
    X_list: list,
    projected_X: np.ndarray,
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
        max_iter = 5
        manifold = Stiefel(mat.shape[0], projected_X.shape[1])
        a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(projected_X[i, :, :, :])

        @pymanopt.function.autograd(manifold)
        def objective_function(proj):
            a_mat_recon = anp.einsum("ba,dc,acg->bdg", proj, proj, a_lhs)
            return anp.sum(anp.square(a_mat - a_mat_recon))

        problem = Problem(manifold=manifold, cost=objective_function)

        # Solve the problem
        solver = TrustRegions(
            verbosity=0, min_gradient_norm=1e-9, min_step_size=1e-12
        )
        proj = solver.run(problem).point

        U, _, Vt = np.linalg.svd(proj, full_matrices=False)
        proj = U @ Vt
        
        # Check if the projection matrix is correct
        A = X_list[i]
        B = project_data(A, proj)
        
        # While this projection matrix is not solved rerun the solver till max_iter
        while not np.allclose(B, projected_X[i, :, :, :]) and max_iter > 0:
            proj = solver.run(problem).point
            U, _, Vt = np.linalg.svd(proj, full_matrices=False)
            proj = U @ Vt
            A = X_list[i]
            B = project_data(A, proj)
            max_iter -= 1
        
        projections.append(proj)

    return projections
