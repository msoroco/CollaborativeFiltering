import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


###   min Trace(Y) = Trace(W1) + Trace(W2)
###        ----------
###    Y = | W1   R |
###        | R^T  W2|
###        ----------

def minimize_trace(X, sample_indices):
    (n, d) = X.shape
    Y = cp.Variable((d+n,d+n),PSD=True) # Require Y to be positive semidefinite
    R = Y[:n,n:]

    obj = cp.Minimize(cp.trace(Y))

    constraints = [] 
    # training indices must match
    for I in sample_indices:
        i = int(np.floor(I/d))
        j = int(I % d)
        constraints.append(R[i,j] == X[i,j])

    prob = cp.Problem(obj,constraints)
    solution = prob.solve()

    return R.value