import numpy as np

def norm(v: np.ndarray, p: int = 2) -> float:
    if p == np.inf:
        return float(np.max(np.abs(v)))
    else:
        return float(np.sum(np.abs(v)**p)**(1/p))

def mnorm(A: np.ndarray, p: int = 2) -> float:
    pass
