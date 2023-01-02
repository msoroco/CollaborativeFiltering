import numpy as np


def gradient_descent(objfn, W, Z, maxItr=1000, epsilon=1e-2, verbose=False ):
    # objfn: function that returns (objective,gradient)
	# W, Z: initial guesses for W, Z respectively
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this

	# Evaluate the intial objective and gradient
    f, g_w, g_z = objfn(W, Z)

    # Set initial step-size so update has an L1-norm at most 1
    alpha = min(1, 1/np.linalg.norm(g_w, ord=1), 1/np.linalg.norm(g_z, ord=1))

    for i in range(maxItr):
       # Try out the current step-size
        w_New = W - alpha*g_w
        z_New = Z - alpha*g_z
        f_New, g_w_New, g_z_New = objfn(w_New, z_New)

        # Decrease the step-size if we increased the function
        while f_New > f:
            alpha = alpha/2
        
            # Try out the smaller step-size
            w_New = W - alpha*g_w
            z_New = Z - alpha*g_z
            f_New, g_w_New, g_z_New = objfn(w_New, z_New)

        # Accept the new parameters/function/gradient
        W = w_New
        Z = z_New
        f = f_New
        g_w = g_w_New
        g_z = g_z_New

        # Print out some diagnostics
        gradNorm = max(np.linalg.norm(g_w.flatten(),ord=np.inf), np.linalg.norm(g_z.flatten(),ord=np.inf))

        # We want to stop if the gradient is really small
        if gradNorm < epsilon:
            if verbose:
                print("Problem solved up to optimality tolerance\n")
            return W, Z
    
    if verbose:
        print("Reached maximum number of iterations\n")
    return W, Z
