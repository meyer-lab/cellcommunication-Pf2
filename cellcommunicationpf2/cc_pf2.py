import autograd.numpy as anp
import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from typing import Optional
from sklearn.utils.extmath import randomized_svd
import os
from copy import deepcopy
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import parafac
import cupy as cp


def TEMP_TODO_factors_to_tensor(factors: list[np.ndarray]) -> np.ndarray:
    pass
    # This will be filled in along with the recon error code


def TEMP_TODO_reconstruction_error(
    factors: list[np.ndarray],
    projected_X: np.ndarray
) -> float:
    pass
    # This will be filled in along with the recon error code


def flatten_tensor_list(tensor_list: list):
    """
    Flatten a list of 3D tensors from A x B x B x C to a matrix of (A*B*B) x C
    """

    # Reshape each tensor to a 2D matrix
    # This will stack rows of each B x B tensor into a single row
    reshaped_tensors = [tensor.reshape(-1, tensor.shape[-1]) for tensor in tensor_list]

    # Vertically stack these matrices
    flattened_matrix = np.vstack(reshaped_tensors)

    return flattened_matrix


def init(
    X_list: list,
    rank: int,
    random_state: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Initializes the factors for the CP decomposition of a list of 3D tensors
    """
    data_matrix = flatten_tensor_list(X_list)
    
    _, _, C = randomized_svd(data_matrix, rank, random_state=random_state)
    factors = [np.ones((len(X_list), rank)), np.eye(rank), np.eye(rank), C.T]
    return factors


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
    Takes a list of 3D tensors of C x C x LR (where each C is different) and an 
    aligned tensor of A x CES x CES x LR and solves for the projection matrices 
    for each tensor as well as reconstruct the data based on the projection matrices.
    """
    projections: list[np.ndarray] = []
    projected_X = np.zeros_like(full_tensor)

    for i, mat in enumerate(X_list):
        manifold = Stiefel(mat.shape[0], full_tensor.shape[1])
        a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(full_tensor[i, :, :, :])

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

        projections.append(proj)
        projected_X[i] = project_data(full_tensor[i], proj)

    return projections, projected_X

def mock_pf_method(
    X_in,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
) -> tuple[tuple, float]:
    """
    Mockup of the PF method for cell-cell communication
    """

    verbose = "CI" not in os.environ

    gamma = 1.1
    gamma_bar = 1.03
    eta = 1.5
    beta_i = 0.05
    beta_i_bar = 1.0

    factors = init(X_in, rank, random_state=random_state)
    TEMP_TODO_norm_tensor = None 
    old_factors = deepcopy(factors)

    projection, projected_X = solve_projections(X_in, TEMP_TODO_factors_to_tensor(factors))

    err = TEMP_TODO_reconstruction_error(factors, projected_X)
    errs = [err]

    print("")
    tq = tqdm(range(n_iter_max), disable=(not verbose))

    for iteration in tq:
        jump = beta_i + 1.0

        # Estimate error with line search
        factors_ls = [
            factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
        ]

        projections_ls, projected_X_ls = solve_projections(X_in, TEMP_TODO_factors_to_tensor(factors))
        err_ls = TEMP_TODO_reconstruction_error(
            factors_ls, projected_X_ls
        )

        if err_ls < errs[-1] * TEMP_TODO_norm_tensor:
            err = err_ls
            projections = projections_ls
            projected_X = projected_X_ls
            factors = factors_ls

            beta_i = min(beta_i_bar, gamma * beta_i)
            beta_i_bar = max(1.0, gamma_bar * beta_i_bar)
        else:
            beta_i_bar = beta_i
            beta_i = beta_i / eta

            projections, projected_X = solve_projections(X_in, TEMP_TODO_factors_to_tensor(factors))
            err = TEMP_TODO_reconstruction_error(factors, projected_X)

        errs.append(err / TEMP_TODO_norm_tensor)

        tl.set_backend("cupy")
        factors_old = deepcopy(factors)
        _, factors = parafac(
            cp.array(projected_X),  # type: ignore
            rank,
            n_iter_max=20,
            init=(None, [cp.array(f) for f in factors]),  # type: ignore
            tol=None,  # type: ignore
            normalize_factors=False,
        )
        tl.set_backend("numpy")
        factors = [cp.asnumpy(f) for f in factors]

        delta = errs[-2] - errs[-1]
        tq.set_postfix(
            error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False
        )

        if delta < tol:
            break

    R2X = 1 - errs[-1]
    # return standardize_pf2(factors, projections), R2X
    return (factors, projections), R2X
