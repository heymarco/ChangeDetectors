import numpy as np
import ot
from ot import bregman


def bonferroni_correction(delta: float, n: int) -> float:
    return delta / n


def sinkhorn_approx(s1, s2, reg=1e-1):
    n1 = len(s1)
    n2 = len(s2)

    a = np.ones((n1,)) / n1
    b = np.ones((n2,)) / n2
    M = ot.dist(s1, s2)

    loss_mat = bregman.sinkhorn_log(a, b, M, reg, verbose=False)
    dist = np.sum(M * loss_mat)
    return dist