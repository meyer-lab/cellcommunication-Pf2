import anndata
import numpy as np
import cupy as cp
import tensorly as tl


def convert_4d_to_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a b^2 x X^2 matrix to a b x X matrix.
    """
    b = int(np.sqrt(matrix.shape[0]))
    X_dim = int(np.sqrt(matrix.shape[1]))

    # Reshape to 4D and average
    reshaped = matrix.reshape(b, b, X_dim, X_dim) # maybe redo this rehsape manually
    return np.mean(reshaped, axis=(1, 3))


def project_tensor(tensor: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D tensor of C x C x LR with a projection matrix of C x CES
    along both C dimensions to form a resulting tensor of CES x CES x LR.
    """

    tensor = np.tensordot(tensor, proj_matrix.T, axes=(1, 0))  # C × CES × LR

    tensor = np.tensordot(proj_matrix, tensor, axes=(1, 0))  # CES × CES × LR

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

    full_tensor = tl.cp_tensor.cp_to_tensor([A, B, C, D])

    for i, mat in enumerate(X_list):
        if isinstance(mat, np.ndarray):
            mat = cp.array(mat)

        lhs = full_tensor[i, :, :, :]
        lhs = lhs.reshape(lhs.shape[0] * lhs.shape[1], lhs.shape[2])
        lhs = lhs.T

        U, _, Vh = cp.linalg.svd(mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2]) @ lhs - means @ lhs, full_matrices=False)
        proj = U @ Vh
        proj = convert_4d_to_2d(
            cp.asnumpy(proj)
        )  # Perform the conversion here since we expect that
        projections.append(proj) 

        # Account for centering (currently not completed)
        centering = cp.outer(cp.sum(proj, axis=0), means)
        projected_X[i, :, :, :] = project_tensor(mat, proj) #- centering # unflatten mat and then store projectedX with an extra dimension to store the full tensor

    return projections, cp.asnumpy(projected_X)
