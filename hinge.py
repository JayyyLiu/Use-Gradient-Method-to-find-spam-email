from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
    [d,n]=np.shape(xTr)
    i=0
    j=0
    loss=0
    gradient=np.zeros((d,1))
    while i<n:
        sum1=max(1-yTr[0,i]*np.dot(np.transpose(w),xTr[:,i:i+1]),0)
        i=i+1
        loss=loss+sum1
    loss=loss+lambdaa*np.dot(np.transpose(w),w)
    while j<n:
        if 1-yTr[0,j]*np.dot(np.transpose(w),xTr[:,j:j+1])>0:
            sum2=-yTr[0,j]*xTr[:,j:j+1]
            j=j+1
            gradient=sum2+gradient
            
        else:
            sum2=np.zeros((d,1))
            j=j+1
            gradient=sum2+gradient
    gradient=gradient+2*lambdaa*w
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE

    return loss,gradient
