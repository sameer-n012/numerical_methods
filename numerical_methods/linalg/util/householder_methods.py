import numpy as np
from .norm_methods import norm

def householder(x, y, w_only=False):

    w = x - y
    if w_only:
        return w/norm(w)

    return np.eye(len(x)) - 2*np.matmul(w, w.T)
