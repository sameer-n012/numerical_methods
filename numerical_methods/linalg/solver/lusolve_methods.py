from numpy import matmul, ndarray, shape
from ..factoring import lu
from . import trisolve

def lusolve(A: ndarray, b: ndarray) -> ndarray:
    """
    Solves a system of equations Ax = b to find the
    exact solution x. Uses LU factorization to solve
    (see method lu() for details). Takes O(m^3) time for an
    m*m matrix.

    Parameters
    ----------
    A : the m*m matrix in the linear system of equations
        Ax = b, a numpy ndarray.
    b : the m*1 matrix in the linear system of equations
        Ax = b, a numpy ndarray.

    Returns
    -------
    x : the m*1 exact solution to the system of equations
        Ax = b, a numpy ndarray.

    Throws
    ------
    ValueError: if the dimensions of A and b are not
                correct.
    """

    m, n = shape(A)

    if m != shape(b)[0] or m != n:
        raise ValueError("Invalid matrix dimensions")

    L, U, P, _, _ = lu(A, pivoting="partial")
    b = matmul(P, b)

    x2 = trisolve(L, b, upper=False)
    x = trisolve(U, x2, upper=True)
    return x
