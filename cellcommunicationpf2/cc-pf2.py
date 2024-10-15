import anndata
import numpy as np
import cupy as cp
import tensorly as tl

def convert_4d_to_list(X: np.ndarray) -> list[np.ndarray]:
    """
    Converts a 4D sc x rc x LR x obs tensor to a list of sc*rc x LR matrices
    where each matrix is a different observation.
    """
    result = []
    for i in range(X.shape[3]):
        result.append(X[:, :, :, i].reshape(-1, X.shape[2]))
    return result

def convert_4d_to_2d(X: list[np.ndarray]) -> np.ndarray:
    """
    Converts a list of b^2 x X^2 matrices to a list of b x X matrices.
    Essentially reshapes each b^2 x X^2 matrix into a 4D b x b x X x X tensor
    and then averages over one of the b dimensions and one of the X dimensions.
    """
    result = []
    for matrix in X:
        b = int(np.sqrt(matrix.shape[0]))
        X_dim = int(np.sqrt(matrix.shape[1]))
        
        # Reshape to 4D and average
        reshaped = matrix.reshape(b, b, X_dim, X_dim)
        averaged = np.mean(reshaped, axis=(1, 3))
        
        result.append(averaged)
    
    return np.array(result)

def project_data(
    X_list: list, means: np.ndarray, factors: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray]:
    A, B, C = factors

    projections: list[np.ndarray] = []
    projected_X = cp.empty((A.shape[0], B.shape[0], C.shape[0])) # Having trouble understanding how to manipulate this for the 4th dimension
    means = cp.array(means)

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat)

        lhs = cp.array((A[i] * C) @ B.T, copy=False)
        U, _, Vh = cp.linalg.svd(mat @ lhs - means @ lhs, full_matrices=False)
        proj = U @ Vh

        projections.append(convert_4d_to_2d(cp.asnumpy(proj))) # Perform the conversion here since we expect that

        # Account for centering
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :] = proj.T @ mat - centering

    return projections, cp.asnumpy(projected_X)


def parafac2_init(
    X_in: anndata.AnnData,
    rank: int,
    random_state: Optional[int] = None,
) -> tuple[list[np.ndarray], float]:
    pass
    # Index dataset to a list of conditions
    # n_cond = len(X_in.obs["condition_unique_idxs"].cat.categories)
    # means = X_in.var["means"].to_numpy()

    # lmult = X_in.X @ means
    # if isinstance(X_in.X, np.ndarray):
    #     norm_tensor = float(np.linalg.norm(X_in.X) ** 2.0 - 2 * np.sum(lmult))
    # else:
    #     norm_tensor = float(norm(X_in.X) ** 2.0 - 2 * np.sum(lmult))

    # _, _, C = randomized_svd(X_in.X, rank, random_state=random_state)  # type: ignore

    # factors = [np.ones((n_cond, rank)), np.eye(rank), C.T]
    # return factors, norm_tensor


def parafac2_nd(
    X_in: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    SECSI_solver=False,
    callback: Optional[Callable[[int, float, list, list], None]] = None,
) -> tuple[tuple, float]:
    r"""The same interface as regular PARAFAC2."""
    pass
    # Verbose if this is not an automated build
    # verbose = "CI" not in os.environ

    # gamma = 1.1
    # gamma_bar = 1.03
    # eta = 1.5
    # beta_i = 0.05
    # beta_i_bar = 1.0

    # factors, norm_tensor = parafac2_init(X_in, rank, random_state)
    # factors_old = deepcopy(factors)

    # X_list = anndata_to_list(X_in)

    # if "means" in X_in.var:
    #     means = np.array(X_in.var["means"].to_numpy())
    # else:
    #     means = np.zeros((1, factors[2].shape[0]))

    # projections, projected_X = project_data(X_list, means, factors)
    # err = reconstruction_error(factors, projections, projected_X, norm_tensor)
    # errs = [err]

    # if SECSI_solver:
    #     SECSerror, factorOuts = SECSI(projected_X, rank, verbose=False)
    #     factors = factorOuts[np.argmin(SECSerror)].factors

    # print("")
    # tq = tqdm(range(n_iter_max), disable=(not verbose))
    # for iteration in tq:
    #     jump = beta_i + 1.0

    #     # Estimate error with line search
    #     factors_ls = [
    #         factors_old[ii] + (factors[ii] - factors_old[ii]) * jump for ii in range(3)
    #     ]

    #     projections_ls, projected_X_ls = project_data(X_list, means, factors)
    #     err_ls = reconstruction_error(
    #         factors_ls, projections_ls, projected_X_ls, norm_tensor
    #     )

    #     if err_ls < errs[-1] * norm_tensor:
    #         err = err_ls
    #         projections = projections_ls
    #         projected_X = projected_X_ls
    #         factors = factors_ls

    #         beta_i = min(beta_i_bar, gamma * beta_i)
    #         beta_i_bar = max(1.0, gamma_bar * beta_i_bar)
    #     else:
    #         beta_i_bar = beta_i
    #         beta_i = beta_i / eta

    #         projections, projected_X = project_data(X_list, means, factors)
    #         err = reconstruction_error(factors, projections, projected_X, norm_tensor)

    #     errs.append(err / norm_tensor)

    #     tl.set_backend("cupy")
    #     factors_old = deepcopy(factors)
    #     _, factors = parafac(
    #         cp.array(projected_X),  # type: ignore
    #         rank,
    #         n_iter_max=20,
    #         init=(None, [cp.array(f) for f in factors]),  # type: ignore
    #         tol=None,  # type: ignore
    #         normalize_factors=False,
    #     )
    #     tl.set_backend("numpy")
    #     factors = [cp.asnumpy(f) for f in factors]

    #     delta = errs[-2] - errs[-1]
    #     tq.set_postfix(
    #         error=errs[-1], R2X=1.0 - errs[-1], Δ=delta, jump=jump, refresh=False
    #     )
    #     if callback is not None:
    #         callback(iteration, errs[-1], factors, projections)

    #     if delta < tol:
    #         break

    # R2X = 1 - errs[-1]
    # return standardize_pf2(factors, projections), R2X
