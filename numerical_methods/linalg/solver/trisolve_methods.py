from numpy import ndarray, shape, sum, zeros

def trisolve(A: ndarray, b: ndarray, upper: bool = True) -> ndarray:
    """
    TODO add description to trisolve method
    """

    m, n = shape(A)

    if m != shape(b)[0]:
        raise ValueError("Invalid matrix dimensions")

    if upper:
        return _utrisolve(A, b)
    else:
        return _ltrisolve(A, b)

def _utrisolve(A, b):

    m, n = shape(A)
    x = zeros((n, 1))

    for col in range(n):
        tmp = b[col] - sum(x*A[col, :].T)
        x[col] = tmp/A[col, col]

    return x

def _ltrisolve(A, b):

    m, n = shape(A)
    x = zeros((n, 1))

    for col in range(n-1, -1, -1):
        tmp = b[col] - sum(x*A[col, :].T)
        x[col] = tmp/A[col, col]

    return x
