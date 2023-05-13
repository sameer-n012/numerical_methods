from numpy import ndarray, ones, zeros, eye, shape, matmul, all
from ..util import norm
from ..util import householder
from ..solver import trisolve

def qr(A: ndarray, verbose: bool = False, signs:ndarray = None) -> (ndarray, ndarray):
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

    R = A
    Q = eye(m)

    for col in range(n):

        x = R[:, col].reshape(m, 1)

        if all(x[col+1:] == 0):
            continue

        y = zeros((m, 1))
        if col != 0:
            y[0:col-1] = x[0:col-1]
        y[col] = signs[col] * norm(x[col:])

        w = householder(x, y, w_only=True)

        proj_Q = matmul(2*matmul(Q[:, col:], w[col:]), w[col:].T)
        Q[:, col:] = Q[:, col:] - proj_Q

        proj_R = matmul(2*w[col:], matmul(w[col:].T, R[col:, col]))
        R[col:, col:] = R[col:, col:] - proj_R

        if verbose:
            print("Q%d".format(col+1), Q)
            print("R%d".format(col+1), R)

    return Q, R


def qrsolve(A: ndarray, b: ndarray, overdetermined: bool = True) -> ndarray:
    """
    TODO : add description to qr solve function
    """

    if overdetermined:
        return _odlsqrsolve(A, b)
    else:
        return _udlsqrsolve(A, b)


def _odlsqrsolve(A, b):

    m, n = shape(A);

    if m != shape(b)[0]:
        raise ValueError("Invalid matrix dimensions")

    R = A

    for col in range(n):

        x = R[:, col]

        if all(x[col+1:] == 0):
            continue

        y = zeros((m, 1))
        if col != 0:
            y[0:col-1] = x[0:col-1]
        y[col] = norm(x[col:])

        w = householder(x, y, w_only=True)
        b = b - matmul(w, 2*matmul(w.T, b))

        proj_R = matmul(2*w[col:], matmul(w[col:].T, R[col:, col:]))
        R[col:, col:] = R[col:, col:] - proj_R

    return trisolve(R[:col, :col], b, upper=True)


def _udlsqrsolve(A, b):
    """
    TODO udls qr solver function
    """

    m, n = shape(A);

    if m != shape(b)[0]:
        raise ValueError("Invalid matrix dimensions")


