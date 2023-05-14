from numpy import ndarray, ones, zeros, eye, shape, matmul, all, double
from ..util import norm, householder

def qr(A: ndarray, verbose: bool = False, signs: ndarray = None) -> (ndarray, ndarray):
    """
    QR-factors the matrix A into upper triangular and
    orthogonal components. Takes O(n^3) time for an m*n
    matrix. Works only where m >= n.

    Parameters
    ----------
    A : the m*n matrix to QR factor, a numpy ndarray
    verbose : whether to print the intermediate Q and R matrices
                as they are being generated. Prints if verbose = True,
                a boolean
    signs : a vector of length n containing the signs of the diagonal
            elements in R, a numpy ndarray

    Returns
    -------
    Q, R : the upper triangular matrix R and the orthogonal
            matrix Q such that Q*R = A. Both are numpy ndarrays

    Throws
    ------
    ValueError : if m < n for the m*n matrix A or the shape of
                    signs is not n*1.
    """

    m, n = shape(A)

    if m < n:
        raise ValueError("Invalid matrix dimensions")

    if signs is None:
        signs = ones((n, 1))

    if shape(signs) != (n, 1):
        raise ValueError("Invalid matrix dimensions")

    R = A.astype(double, copy=True)
    Q = eye(m).astype(double, copy=False)

    for col in range(n):

        x = R[0:m, col:col+1]

        if all(x[col+1:m] == 0):
            continue

        y = zeros((m, 1)).astype(double, copy=False)
        if col != 0:
            y[0:col-1] = x[0:col-1]
        y[col] = signs[col] * norm(x[col:m])

        w = householder(x, y, w_only=True)

        proj_Q = matmul(2*matmul(Q[0:m, col:m], w[col:m]), w[col:m].T)
        Q[0:m, col:m] = Q[0:m, col:m] - proj_Q

        proj_R = matmul(2*w[col:m], matmul(w[col:m].T, R[col:m, col:n]))
        R[col:m, col:n] = R[col:m, col:n] - proj_R

        if verbose:
            print("Q{}".format(col+1), Q)
            print("R{}".format(col+1), R)

    return Q, R
