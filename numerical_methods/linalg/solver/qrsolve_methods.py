from numpy import ndarray, zeros, shape, matmul, all
from ..util import norm, householder
from . import trisolve


def qrsolve(A: ndarray, b: ndarray, overdetermined: bool = True) -> ndarray:
    """
    Solves a system of equations Ax = b to find the
    least-squares solution x. Uses QR factorization to solve
    (see method qr() for details). Takes O(m^3) time for an
    m*n matrix.

    Parameters
    ----------
    A : the m*n matrix in the linear system of equations
        Ax = b, a numpy ndarray.
    b : the m*1 matrix in the linear system of equations
        Ax = b, a numpy ndarray.
    overdetermined : whether the system is overdetermined
                        (True) or underdetermined (False), a
                        boolean.

    Returns
    -------
    x : the n*1 least-squares solution to the system of
        equations Ax = b, a numpy ndarray.

    Throws
    ------
    ValueError: if the dimensions of A and b are not
                correct.
    """

    if overdetermined:
        return _odlsqrsolve(A, b)
    else:
        return _udlsqrsolve(A, b)


def _odlsqrsolve(A, b):

    m, n = shape(A)

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

    return trisolve(R[0:n, 0:n], b, upper=True)


def _udlsqrsolve(A, b):
    """
    TODO udls qr solver function
    """

    raise NotImplementedError()
