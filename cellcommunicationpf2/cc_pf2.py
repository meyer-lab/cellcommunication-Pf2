import anndata
import numpy as np
import pymanopt
import sparse
from opt_einsum import contract
from pacmap import PaCMAP
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.tenalg.svd import randomized_svd
from parafac2.parafac2 import parafac2_nd
from cellcommunicationpf2.ccc import calc_communication_score
import pandas as pd


def reconstruction_error(
    factors: list[np.ndarray], original_X: np.ndarray, projections: list[np.ndarray]
) -> float:
    """
    Compute the reconstruction error of the CP decomposition
    """
    reconstructed_X = cp_to_tensor((None, factors))

    recon_err = 0.0

    for i, (orig_tensor, proj) in enumerate(zip(original_X, projections, strict=False)):
        projected_X = project_data(reconstructed_X[i, :, :, :], proj.T)

        recon_err += float(
            np.linalg.norm(sparse.asnumpy(orig_tensor) - sparse.asnumpy(projected_X))
            ** 2
        )

    return recon_err


def init(
    X_list: list,
    rank: int,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """
    Initializes the factors for the CP decomposition of a list of 3D tensors
    """
    # Reshape each tensor to a 2D matrix
    # This will stack rows of each B x B tensor into a single row
    flattened_tensors = [t.reshape((-1, t.shape[2])) for t in X_list]

    if isinstance(flattened_tensors[0], sparse.SparseArray):
        data_matrix = sparse.concatenate(flattened_tensors, 0)
    else:
        data_matrix = np.concatenate(flattened_tensors, 0)

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
    random_seed: int | None = None,
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

    # Solve the problem
    solver = ConjugateGradient(
        verbosity=0, min_gradient_norm=1e-10, min_step_size=1e-12
    )

    for i, tensor in enumerate(X_list):
        manifold = Stiefel(tensor.shape[0], full_tensor.shape[1])

        tensor = sparse.asnumpy(tensor)

        # LHS full slice
        a_lhs = full_tensor[i, :, :, :]
        a_lhs = sparse.asnumpy(a_lhs)

        # Generate a reproducible initial point on the Stiefel manifold
        X = rng.randn(tensor.shape[0], full_tensor.shape[1])
        # Use QR factorization to get a point on the Stiefel manifold
        initial_point, _ = np.linalg.qr(X)

        # Term 1: ||tensor||^2 - constant term, precomputed outside this
        # function for efficiency as the function is called many times
        tensor_squared_norm = np.linalg.norm(tensor)

        @pymanopt.function.numpy(manifold)
        def projection_loss_function(proj):
            """
            Computes the projection loss without creating the large a_mat_recon tensor.
            Uses the expansion of ||tensor - a_mat_recon||^2 = ||tensor||^2 -
                2<tensor, a_mat_recon> + ||a_mat_recon||^2
            """
            proj_a_lhs = contract("ba,acg->bcg", proj, a_lhs)

            # Term 2: -2<tensor, a_mat_recon>
            # Compute the inner product without creating the full a_mat_recon
            inner_product = contract("bdg,bcg,dc->", tensor, proj_a_lhs, proj)

            # Term 3: ||a_mat_recon||^2
            # Compute the squared norm of a_mat_recon without creating the full tensor
            recon_squared_norm = contract(
                "ba,dc,bcg,dcg->", proj, proj, proj_a_lhs, proj_a_lhs
            )

            # Combine all terms
            return tensor_squared_norm - 2 * inner_product + recon_squared_norm

        @pymanopt.function.numpy(manifold)
        def projection_gradient_function(proj):
            """
            Computes the Euclidean gradient of the projection_loss_function
            with respect to the projection matrix proj.
            """
            # Calculate the reconstructed tensor
            # a_mat_recon has shape (N, N, G)
            a_mat_recon = contract("ba,dc,acg->bdg", proj, proj, a_lhs)
            # Calculate the error tensor E = a_mat - a_mat_recon
            # E has shape (N, N, G)
            E = tensor - a_mat_recon
            # Calculate the two terms of the gradient using einsum
            # proj has shape (N, M), a_lhs has shape (M, M, G)
            # grad_term1 corresponds to differentiating wrt the first proj ('ba')
            # grad_term1 = sum_{d,g,c} E_{bdg} * proj_{dc} * a_lhs_{acg}
            # Output shape should be (N, M) matching proj ('ba')
            grad_term1 = contract("bdg,dc,acg->ba", E, proj, a_lhs)
            # grad_term2 corresponds to differentiating wrt the second proj ('dc')
            # grad_term2 = sum_{b,g,a} E_{bdg} * proj_{ba} * a_lhs_{acg}
            # Output shape should be (N, M) matching proj ('dc')
            grad_term2 = contract("bdg,ba,acg->dc", E, proj, a_lhs)
            # Combine the terms and scale by -2
            gradient = -2 * (grad_term1 + grad_term2)
            return gradient

        problem = Problem(
            manifold=manifold,
            cost=projection_loss_function,
            euclidean_gradient=projection_gradient_function,
        )

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
    random_state: int | None = None,
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


def fit_cc_pf2(
    X: anndata.AnnData,
    rank: int,
    random_state=1,
    do_embedding: bool = True,
    tol=1e-7,
    max_iter: int = 100,
) -> tuple[anndata.AnnData, float]:
    """
    Fits the Pf2 decomposition for a list of 3D tensors
    """
    cc_pf2_out, r2x = cc_pf2(
        X, rank=rank, random_state=random_state, tol=tol, n_iter_max=max_iter
    )

    data = store_cc_pf2(X, cc_pf2_out)

    if do_embedding:
        pcm = PaCMAP(random_state=random_state)
        data.obsm["Pf2_PaCMAP_projections"] = pcm.fit_transform(
            data.obsm["Pf2_cell_cell_projections"]
        )  # type: ignore

    return data, r2x


def temp_calc_communication_score(ces_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate cell-cell interaction scores for a given CES matrix.
    This function is a temporary placeholder and should be replaced with
    the actual implementation of calc_communication_score.
    """
    # Placeholder for the actual calculation logic. Will currently just transform the matrix to a 3D tensor by filling the third dimension with random values.
    # Replace this with the actual implementation of calc_communication_score.
    rank = ces_matrix.shape[0]
    genes = ces_matrix.shape[1]
    # Create a random tensor of shape (rank, rank, genes) to simulate the interaction
    interaction_tensor = np.random.rand(rank, rank, genes)
    return interaction_tensor


def cc_pf2_redesigned(
    X_list: list[np.ndarray],
    rank: int,
    n_iter_max: int,
    tol: float,
    random_state: int | None = None,
) -> tuple[tuple, float]:
    """
    Redesigned cell-cell communication model using initial PARAFAC2
    followed by CP decomposition
    """
    import pandas as pd
    import anndata

    # Convert the X_list into an anndata object that can be used with parafac2_nd
    n_conditions = len(X_list)
    n_features = X_list[0].shape[1]  # genes dimension

    # Calculate total number of cells across all conditions
    total_cells = sum(x.shape[0] for x in X_list)

    # Concatenate all matrices to create the full data matrix
    X_full = np.vstack(X_list)

    # Create condition indices for each cell
    condition_idxs = []
    for i, x in enumerate(X_list):
        condition_idxs.extend([i] * x.shape[0])

    # Create observation dataframe
    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(total_cells)])
    obs_df["condition_unique_idxs"] = condition_idxs

    # Create variable dataframe
    var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_features)])

    # Add small amount of noise to avoid perfect rank-deficiency
    if random_state is not None:
        np.random.seed(random_state)
    X_full = X_full + np.random.normal(0, 1e-6, X_full.shape)

    # Create the AnnData object
    adata = anndata.AnnData(X=X_full, obs=obs_df, var=var_df)

    # Call parafac2_nd with our constructed AnnData
    pf2_output, pf2_r2x = parafac2_nd(
        adata, rank=rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )

    # Unpack results
    weights, factors, projections = pf2_output

    # Step 2: Project each matrix down to standardized dimension
    projected_tensors = []
    for i, tensor in enumerate(X_list):
        proj = projections[i]
        projected_tensors.append(proj.T @ tensor)  # (rank x genes)

    # Step 3: Calculate cell-cell interaction scores for each sample
    interaction_tensors = []
    for i, ces_matrix in enumerate(projected_tensors):
        # This creates (rank x rank x genes) tensors from (rank x genes) matrices
        interaction_tensor = temp_calc_communication_score(ces_matrix)
        interaction_tensors.append(interaction_tensor)

    interaction_tensors = np.stack(interaction_tensors)

    # Step 4: Run standard CP decomposition on the interaction tensors
    _, cp_factors = parafac(
        interaction_tensors,
        rank,
        n_iter_max=20,
        tol=None,
        normalize_factors=False,
    )

    # Calculate final R2X directly without using reconstruction_error
    reconstructed = cp_to_tensor((None, cp_factors))

    # Calculate total variance and error
    total_variance = np.sum(interaction_tensors**2)
    error = np.sum((interaction_tensors - reconstructed) ** 2)

    # Calculate R2X (1 - normalized error)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return (cp_factors, projections), final_R2X


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


def store_cc_pf2(X: anndata.AnnData, parafac2_output: tuple):
    """Store the Pf2 results into the anndata object."""

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.uns["Pf2_C"], X.varm["Pf2_D"] = parafac2_output[1]
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
