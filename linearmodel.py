import numpy as np

def linearmodel(w,xTe):
#    [d,n]=np.shape(xTe)
#    w=np.zeros((d,1))
    preds=np.dot(np.transpose(xTe),w)
# INPUT:
# w weight vector (default w=0)
# xTe dxn matrix (each column is an input vector)
#
# OUTPUTS:
#
# preds predictions

    # YOUR CODE HERE

    return preds
