from misc import apply
from gradientDescent import gradient_descent
from functools import partial
import numpy as np



def L2Error(W, Z, X, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W
    m = apply(Xhat, X, sample_indices)

    f = sum(((Xhat - m).flatten())**2)
    g_w = np.zeros(W.shape)
    
    for i in range(n):
        e = ((W.T @ Z[i,:]) - m[i,:] )
        for j in range(d):
            g_w[:,j] = g_w[:,j] + 2 * e[j] * Z[i,:]

    g_z = np.zeros(Z.shape)
    
    for j in range(d):
        e = ((Z @ W[:, j]) - m[:,j] )
        for i in range(n):
            g_z[i,:] = g_z[i,:] + 2 * e[i] * W[:,j]

    return f, g_w, g_z



def L2Error_vectorized(W, Z, X, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W
    m = apply(Xhat, X, sample_indices)
    f = sum(((Xhat - m).flatten())**2)

    g_w = np.zeros(W.shape)
    g_w = g_w.flatten('F')
 
    for i in range(n):
        e = ((W.T @ Z[i,:]) - m[i,:])
        E = np.repeat(e, k)
        S = np.tile(Z[i,:], d)
        g_w = g_w + 2 *(E * S) # element wise multiplication of vectors

    g_w = g_w.reshape(k, d, order='F') 


    g_z = np.zeros(Z.shape)
    g_z = g_z.flatten('C')
    
    for j in range(d):
        e = ((Z @ W[:, j]) - m[:,j])
        E = np.repeat(e, k)
        S = np.tile(W[:,j], n)
        g_z = g_z + 2*(E * S) # element wise multiplication of vectors
    
    g_z = g_z.reshape(n, k, order='C')

    return f, g_w, g_z


def LeastSquares(X, k, sample_indices):
    (n,d) = X.shape
    # Initial guess
    Z = np.random.random((n, k))
    W = np.random.random((k, d))

    # Function we're going to minimize (and that computes gradient)
    objfn = partial(L2Error_vectorized, X = X, sample_indices = sample_indices)

    # Solve least squares problem
    W, Z = gradient_descent(objfn, W, Z, maxItr=1000)

    # Return model
    return W, Z




def L2Error_bias_vectorized(W, Z, X, bias, sample_indices):
    (n,d) = X.shape
    (n,k) = Z.shape

    Xhat = Z@W + bias

    m = apply(Xhat, X, sample_indices)
    f = sum(((Xhat - m).flatten())**2)

    g_w = np.zeros(W.shape)
    g_w = g_w.flatten('F')
 
    for i in range(n):
        e = ((W.T @ Z[i,:]) + bias - m[i,:])
        E = np.repeat(e, k)
        S = np.tile(Z[i,:], d)
        g_w = g_w + 2 *(E * S) # element wise multiplication of vectors

    g_w = g_w.reshape(k, d, order='F') 


    g_z = np.zeros(Z.shape)
    g_z = g_z.flatten('C')
    
    for j in range(d):
        e = ((Z @ W[:, j]) + bias - m[:,j])
        E = np.repeat(e, k)
        S = np.tile(W[:,j], n)
        g_z = g_z + 2*(E * S) # element wise multiplication of vectors
    
    g_z = g_z.reshape(n, k, order='C')

    return f, g_w, g_z


def LeastSquares_bias(X, k, sample_indices):
    (n,d) = X.shape
    # Initial guess
    Z = np.random.random((n, k))
    W = np.random.random((k, d))
    b = np.mean(X.flatten()[sample_indices])

    # Function we're going to minimize (and that computes gradient)
    objfn = partial(L2Error_bias_vectorized, X = X, bias=b, sample_indices = sample_indices)

    # Solve least squares problem
    W, Z = gradient_descent(objfn, W, Z, maxItr=1000)

    # Return model
    return W, Z, b
