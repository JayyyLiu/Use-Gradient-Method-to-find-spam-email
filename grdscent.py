
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    w = w0
    lossNext = 1
    for i in range(maxiter):

        loss, grad = func(w)
        
        if lossNext >= loss:
            stepsize = stepsize * 1.01
        else:
            stepsize = stepsize * 0.5
        
        wNext = w - stepsize * grad
        
        if np.linalg.norm(grad, ord = 1) < tolerance:
            break  
    
        w = wNext
        lossNext = loss
    
    return w
