import autograd.numpy as anp
import numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from typing import Optional
from sklearn.utils.extmath import randomized_svd
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
import scipy.sparse as sp


def reconstruction_error(
    factors: list[np.ndarray], original_X: np.ndarray, projections: list[np.ndarray]
) -> float:
    """
    Compute the reconstruction error of the CP decomposition
    """
    reconstructed_X = cp_to_tensor((None, factors))

    recon_err = 0.0

    for i, (orig_tensor, proj) in enumerate(zip(original_X, projections)):
        projected_X = project_data(reconstructed_X[i, :, :, :], proj.T)
        
        # Get coordinates and data from current sparse tensor
        coords = orig_tensor.coords
        data = orig_tensor.data
        
        # Compare only at non-zero locations
        recon_vals = projected_X[coords[0], coords[1], coords[2]]
        diff = data - recon_vals
        recon_err += (diff ** 2).sum()

    return recon_err


def flatten_tensor_list(tensor_list: list) -> np.ndarray:
    """
    Flatten a list of 3D tensors from A x B x B x C to a matrix of (A*B*B) x C
    """
    
    flattened_tensors = []
    # Reshape each tensor to a 2D matrix
    # This will stack rows of each B x B tensor into a single row
    for tensor in tensor_list:
        # Get tensor properties
        coords = tensor.coords
        data = tensor.data
        shape = tensor.shape
        
        # Calculate new row indices for flattened tensor
        new_rows = coords[0] * shape[1] + coords[1]
        new_cols = coords[2]
        
        # Create flattened sparse matrix
        flat_shape = (shape[0] * shape[1], shape[2])
        flattened = sp.csr_matrix(
            (data, (new_rows, new_cols)),
            shape=flat_shape
        )
        flattened_tensors.append(flattened)
    
    return sp.vstack(flattened_tensors)


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
    along both C dimensions to form a resulting in a 3D tensor of CES x CES x LR.
    """
    return np.einsum("ab,cd,acg->bdg", proj_matrix, proj_matrix, tensor)


def solve_projections(
    X_list: list,
    full_tensor: np.ndarray,
    random_seed: Optional[int] = None,
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

    rng = np.random.RandomState(random_seed)

    for i, mat in enumerate(X_list):
        manifold = Stiefel(mat.shape[0], full_tensor.shape[1])
        # a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(full_tensor[i, :, :, :])
        
        coords = mat.coords
        data = mat.data
        i_idx, j_idx, k_idx = coords[0], coords[1], coords[2]

        # Generate a reproducible initial point on the Stiefel manifold
        X = rng.randn(mat.shape[0], full_tensor.shape[1])
        # Use QR factorization to get a point on the Stiefel manifold
        Q, _ = np.linalg.qr(X)
        initial_point = Q

        @pymanopt.function.autograd(manifold)
        def projection_loss_function(proj):
            a_mat_recon = anp.einsum("ba,dc,acg->bdg", proj, proj, a_lhs)
            
            recon_vals = a_mat_recon[j_idx, i_idx, k_idx]
            
            return anp.sum(anp.square(data - recon_vals))

        problem = Problem(
            manifold=manifold,
            cost=projection_loss_function,
        )

        # Solve the problem
        solver = TrustRegions(verbosity=0, min_gradient_norm=1e-9, min_step_size=1e-12)

        proj = solver.run(problem, initial_point=initial_point).point

        U, _, Vt = np.linalg.svd(proj, full_matrices=False)
        proj = U @ Vt

        projections.append(proj)

    return projections


def cc_pf2(
    X_list: list,
    rank: int,
    n_iter_max: int,
    tol: float,
    random_state: Optional[int] = None,
) -> tuple[tuple, float]:
    """
    Fits the factors of the CP decomposition for a list of 3D tensors
    """

    factors = init(X_list, rank, random_state=random_state)
    errs = []

    for i in range(n_iter_max):
        full_tensor = cp_to_tensor((None, factors))
        projections = solve_projections(X_list, full_tensor, random_seed=random_state)
        err = reconstruction_error(factors, X_list, projections)
        errs.append(err)

        projected_X = [
            project_data(X_list[i], proj) for i, proj in enumerate(projections)
        ]
        projected_X = np.stack(projected_X, axis=0)
        _, factors = parafac(
            projected_X,
            rank,
            n_iter_max=20,
            init=(None, [np.array(f) for f in factors]),
            tol=None,
            normalize_factors=False,
        )

        delta = abs(errs[-2] - errs[-1]) if i > 0 else tol + 1

        print(f"Iteration {i}, Error: {err}, Delta: {delta}")

        if delta < tol:
            print("Converged")
            break

    final_err = errs[-1]
    return (factors, projections), final_err
