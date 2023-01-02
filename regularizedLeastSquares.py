from misc import apply
from gradentDescent import gradient_descent
from functools import partial
import numpy as np



def Regularized_L2Error(W, Z, X, lambda_1, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W
    m = apply(Xhat, X, sample_indices)

    f = sum(np.square((Xhat - m).flatten()))

    g_w = 2 * lambda_1 * W
    
    for i in range(n):
        e = ((W.T @ Z[i,:]) - m[i,:] )
        for j in range(d):
            g_w[:,j] = 2 * e[j] * Z[i,:] + 2 * lambda_1 * W[:, j]

    
    g_z = 2 * lambda_1 * Z
    
    for j in range(d):
        e = ((Z @ W[:, j]) - m[:,j])
        for i in range(n):
            g_z[i,:] = g_z[i,:] + 2 * e[i] * W[:,j]

 

    return f, g_w, g_z


def Regularized_L2Error_vectorized(W, Z, X, lambda_1, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W
    m = apply(Xhat, X, sample_indices)
    f = sum(np.square((Xhat - m).flatten()))

    g_w = 2 * lambda_1 * W
    g_w = g_w.flatten('F')
 
    for i in range(n):
        e = ((W.T @ Z[i,:]) - m[i,:])
        E = np.repeat(e, k)
        S = np.tile(Z[i,:], d)
        g_w = g_w + 2 *(E * S) # element wise multiplication of vectors

    g_w = g_w.reshape(k, d, order='F') 


    g_z = 2 * lambda_1 * Z
    g_z = g_z.flatten('C')
    
    for j in range(d):
        e = ((Z @ W[:, j]) - m[:,j])
        E = np.repeat(e, k)
        S = np.tile(W[:,j], n)
        g_z = g_z + 2*(E * S) # element wise multiplication of vectors
    
    g_z = g_z.reshape(n, k, order='C')

    return f, g_w, g_z

def Reg_LeastSquares(X, k, lambda1, sample_indices):
    (n,d) = X.shape
    Z = np.random.random((n, k))
    W = np.random.random((k, d))

    # Function we're going to minimize (and that computes gradient)
    objfn = partial(Regularized_L2Error_vectorized, X = X, lambda_1 = lambda1, sample_indices=sample_indices)

    # Solve least squares problem
    W, Z = gradient_descent(objfn, W, Z)

    # Return model
    return W, Z




def Regularized_L2Error_bias_vectorized(W, Z, X, bias, lambda_1, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W + bias

    m = apply(Xhat, X, sample_indices)
    f = sum(np.square((Xhat - m).flatten()))

    g_w = 2 * lambda_1 * W
    g_w = g_w.flatten('F')
 
    for i in range(n):
        e = ((W.T @ Z[i,:]) + bias - m[i,:])
        E = np.repeat(e, k)
        S = np.tile(Z[i,:], d)
        g_w = g_w + 2 *(E * S) # element wise multiplication of vectors

    g_w = g_w.reshape(k, d, order='F') 


    g_z = 2 * lambda_1 * Z
    g_z = g_z.flatten('C')
    
    for j in range(d):
        e = ((Z @ W[:, j]) + bias - m[:,j])
        E = np.repeat(e, k)
        S = np.tile(W[:,j], n)
        g_z = g_z + 2*(E * S) # element wise multiplication of vectors
    
    g_z = g_z.reshape(n, k, order='C')

    return f, g_w, g_z



def Reg_LeastSquares_bias(X, k, lambda1, sample_indices):
    (n,d) = X.shape
    Z = np.random.random((n, k))
    W = np.random.random((k, d))
    b = np.mean(X.flatten()[sample_indices])

    # Function we're going to minimize (and that computes gradient)
    objfn = partial(Regularized_L2Error_bias_vectorized, X = X, bias=b, lambda_1 = lambda1, sample_indices=sample_indices)

    # Solve least squares problem
    W, Z = gradient_descent(objfn, W, Z)

    # Return model
    return W, Z, b