from numpy import ndarray, zeros, eye, shape, matmul, double, argmax, abs, unravel_index, diag, multiply, prod, all


def lu(A: ndarray, pivoting: str = "partial", verbose: bool = False) -> (ndarray, ndarray, ndarray, ndarray, int):
    """
    LU-factors the square matrix A into upper triangular and
    lower triangular components. Takes O(m^3) time for an
    m*m matrix.

    Parameters
    ----------
    A : the m*m matrix to LU factor, a numpy ndarray.
    pivoting : whether to use pivoting to improve the
                stability ot the algorithm. Takes
                    - "none" -> no pivoting
                    - "partial" -> partial pivoting
                    - "full" -> full pivoting
                 , a string.
    verbose : whether to print the intermediate L and U
                matrices as they are being generated. Prints
                if verbose = True, a boolean.

    Returns
    -------
    L, U, P1, P2, s: the upper triangular matrix U and the
                        lower triangular matrix L such that
                        L*U = P1*A*P2 and s column/row
                        swaps were done while pivoting. L,
                        U, P1, P2 are numpy ndarrays and s
                        is an int

    Throws
    ------
    ValueError : if m < n for the m*n matrix A or a pivot
                    value of 0 is encountered
    """

    m, n = shape(A)

    if m < n:
        raise ValueError("Invalid matrix dimensions")

    pivoting = pivoting.lower()
    if pivoting != "none" and pivoting != "partial" and pivoting != "full":
        raise ValueError("Invalid pivoting key")

    U = A.astype(double, copy=True)
    L = zeros((m, m))
    P1 = eye(m)
    P2 = eye(m)
    s = 0

    for i in range(m-1):

        if pivoting == "partial":
            idx = argmax(abs(U[i:m, i:i+1]))

            U = _swaprows(U, i, i + idx)
            P1 = _swaprows(P1, i, i + idx)
            L = _swaprows(L, i, i + idx)

            if idx != 0:
                s += 1

        elif pivoting == "full":
            idx1, idx2 = unravel_index(argmax(abs(U[i:m, i:m])), (m-i, m-i))

            U = _swaprows(U, i, i + idx1)
            P1 = _swaprows(P1, i, i + idx1)
            L = _swaprows(L, i, i + idx1)

            if idx1 != 0:
                s += 1

            U = _swapcols(U, i, i + idx2)
            P1 = _swapcols(P1, i, i + idx2)
            L = _swapcols(L, i, i + idx2)

            if idx2 != 0:
                s += 1

        if U[i, i] == 0:
            raise ValueError("Invalid pivot value")

        l = zeros((m, 1))
        l[i+1:m] = -U[i+1:m, i]/U[i, i]

        U_proj = matmul(l[i+1:m].reshape(m-i-1, 1), U[i:i+1, i:m])
        U[i+1:m, i:m] = U[i+1:m, i:m] + U_proj

        L[:, i] = -l
        L[i, i] = 1

        if verbose:
            print("L{}".format(i), L)
            print("U{}".format(i), U)

    return L, U, P1, P2, s


def ldu(A: ndarray, pivoting: str = "partial", verbose: bool = False) -> (ndarray, ndarray, ndarray,
                                                                          ndarray, ndarray, int):
    """
    LDU-factors the square matrix A into upper triangular
    diagonal, and lower triangular components. Takes O(m^3)
    time for an m*m matrix.

    Parameters
    ----------
    A : the m*m matrix to LU factor, a numpy ndarray.
    pivoting : whether to use pivoting to improve the
                stability ot the algorithm. Takes
                    - "none" -> no pivoting
                    - "partial" -> partial pivoting
                    - "full" -> full pivoting
                 , a string.
    verbose : whether to print the intermediate L and U
                matrices as they are being generated. Prints
                if verbose = True, a boolean.

    Returns
    -------
    L, D, U, P1, P2, s: the upper triangular matrix U,
                        diagonal D, the lower triangular
                        matrix L such that L*U = P1*A*P2 and
                        s column/row swaps were done while
                        pivoting. L, D, U, P1, P2 are numpy
                        ndarrays and s is an int

    Throws
    ------
    ValueError : if m < n for the m*n matrix A or a pivot
                    value of 0 is encountered
    """

    L, U, P1, P2, s = lu(A, pivoting=pivoting, verbose=verbose)
    D, U = _factorutridiag(U)

    return L, D, U, P1, P2, s


def ludet(A: ndarray) -> double:
    """
    Uses LU-factorization to find the determinant of the
    m*m matrix A.

    Parameters
    ----------
    A : the m*m matrix to LU factor, a numpy ndarray.

    Returns
    -------
    d : the determinant of the matrix A

    Throws
    ------
    ValueError : if a pivot value of 0 is encountered
    """

    _, U, _, _, s = lu(A, pivoting="partial")
    d = prod(diag(U))*((-1)**s)
    return double(d)


def luposdef(A: ndarray) -> bool:
    """
    Uses LU-factorization to find if the square, symmetric
    matrix A is positive definite.

    Parameters
    ----------
    A : the m*m matrix to LU factor, a numpy ndarray.

    Returns
    -------
    b : True if A is positive definite, False otherwise,
        a boolean.

    Throws
    ------
    ValueError : if A is not a square matrix or LU-factoring
                    A fails.
    """

    m, n = shape(A)
    if m != n:
        raise ValueError("Invalid matrix dimensions")

    try:
        _, U, _, _, _ = lu(A, pivoting="none")
        b = all(diag(U) > 0)
        return b
    except ValueError as ve:
        if str(ve) != "Invalid pivot value":
            b = False
            return b
        raise ve


def _factorutridiag(A):

    U = A.astype(double, copy=True)
    m, n = shape(U)

    d = diag(U)
    d_inv = 1/d

    for i in range(m-1):
        U[i, i] = 1
        U[i, i+1:m] = multiply(d_inv, U[i, i+1:m])

    U[m-1, m-1] = 1
    D = diag(d)

    return D, U


def _swaprows(A, i, j):
    if i != j:
        tmp = A[i, :]
        A[i, :] = A[j, :]
        A[j, :] = tmp

    return A


def _swapcols(A, i, j):
    if i != j:
        tmp = A[:, i]
        A[:, i] = A[:, j]
        A[:, j] = tmp

    return A
