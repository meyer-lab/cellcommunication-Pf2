import autograd.numpy as anp
import numpy as np
import anndata
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions
from typing import Optional
from sklearn.utils.extmath import randomized_svd
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment

def reconstruction_error(
    factors: list[np.ndarray], original_X: np.ndarray, projections: list[np.ndarray]
) -> float:
    """
    Compute the reconstruction error of the CP decomposition
    """
    reconstructed_X = cp_to_tensor((None, factors))

    recon_err = 0.0

    for i, proj in enumerate(projections):
        projected_X = project_data(reconstructed_X[i, :, :, :], proj.T)
        recon_err += np.linalg.norm(original_X[i] - projected_X) ** 2

    return recon_err


def flatten_tensor_list(tensor_list: list) -> np.ndarray:
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
        a_mat = anp.asarray(mat)
        a_lhs = anp.asarray(full_tensor[i, :, :, :])

        # Generate a reproducible initial point on the Stiefel manifold
        X = rng.randn(mat.shape[0], full_tensor.shape[1])
        # Use QR factorization to get a point on the Stiefel manifold
        Q, _ = np.linalg.qr(X)
        initial_point = Q

        @pymanopt.function.autograd(manifold)
        def projection_loss_function(proj):
            a_mat_recon = anp.einsum("ba,dc,acg->bdg", proj, proj, a_lhs)
            return anp.sum(anp.square(a_mat - a_mat_recon))

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


def fit_pf2(
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
        _, factors = parafac(
            np.array(projected_X),
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



def standardize_cc_pf2(
    factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]

    weights, factors = cp_flip_sign(cp_normalize((None, factors)), mode=1)

    for i in [1, 2]:
        # Order eigen-cells to maximize the diagonal of B/C
        _, col_ind = linear_sum_assignment(np.abs(factors[i].T), maximize=True)
        factors[i] = factors[i][col_ind, :]
        projections = [p[:, col_ind] for p in projections]

        # Flip the sign based on B/C
        signn = np.sign(np.diag(factors[i]))
        factors[i] *= signn[:, np.newaxis]
        projections = [p * signn for p in projections]

    return weights, factors, projections



def store_cc_pf2(
    X: anndata.AnnData,  
    parafac2_output: tuple):
    """Store the Pf2 results into the anndata object."""

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"],  X.uns["Pf2_C"], X.varm["Pf2_D"] = parafac2_output[1]
    projections = parafac2_output[2]

    stacked_projections = np.vstack(projections)

    # Create condition index vector matching stacked projections
    condition_indices = []
    for idx, proj in enumerate(projections):
        condition_indices.extend([idx] * proj.shape[0])

    # Store both stacked projections and their condition indices
    X.uns["Pf2_projections"] = stacked_projections
    X.uns["Pf2_weighted_projections"] = stacked_projections @ X.uns["Pf2_B"]
    X.uns["Pf2_projection_conditions"] = np.array(condition_indices)

    n_pairs = X.shape[0]  # Number of cell-cell pairs
    n_components = len(X.uns["Pf2_weights"])  # Number of components

    # Initialize projection scores matrix
    proj_scores = np.zeros((n_pairs, n_components))
    cell_cell_indices = np.zeros(n_pairs, dtype=int)

    current_idx = 0
    # Process each sample separately
    samples = X.obs["sample"].unique()
    for k in range(len(samples)):
        sample_proj = projections[k] 
        n_cells = sample_proj.shape[0]

        # Generate all cell pairs for this sample
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:  # Skip self-interactions
                    # Calculate projection products and store at current index
                    proj_scores[current_idx, :] = sample_proj[i] * sample_proj[j]
                    cell_cell_indices.append(k)
                    current_idx += 1
                    

    X.obsm["Pf2_cell_cell_projections"] = proj_scores
    X.obsm["Pf2_cell_cell_weighted_projections"] = proj_scores @ X.uns["Pf2_B"]
    X.obsm["Pf2_cell_cell_condition"] = cell_cell_indices

    return X
