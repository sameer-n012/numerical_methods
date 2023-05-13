from numpy import eye, shape, matmul
from .norm_methods import norm

def householder(x, y, w_only=False):

    m = shape(x)[0]

    w = x - y
    if w_only:
        return (w/norm(w)).reshape((m, 1))

    return (eye(m) - 2*matmul(w, w.T)).reshape((m, m))
