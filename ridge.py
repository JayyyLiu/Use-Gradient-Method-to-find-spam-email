
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
    yTr=np.transpose(yTr)
    gradient=2*np.dot(xTr,np.dot(np.transpose(xTr),w)-yTr)+2*lambdaa*w
    loss=np.dot(np.transpose(np.dot(np.transpose(xTr),w)-yTr),np.dot(np.transpose(xTr),w)-yTr)+lambdaa*np.dot(np.transpose(w),w)
    yTr=np.transpose(yTr)
    
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE

    return loss,gradient
