import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):
    [d,n]=np.shape(xTr)
    i=0
    j=0
    loss=0
    gradient=np.zeros((d,1))
    while i < n:
        sum1=math.log(1+math.exp(np.dot(-1*yTr[0,i]*np.transpose(w),xTr[:,i:i+1])))
        i=i+1
        loss=loss+sum1
        
    while j < n :
        sum2=math.exp(np.dot(-1*yTr[0,j]*np.transpose(w),xTr[:,j:j+1]))/(1+math.exp(np.dot(-1*yTr[0,j]*np.transpose(w),xTr[:,j:j+1])))*-1*yTr[0,j]*xTr[:,j:j+1]
        j=j+1
        gradient=gradient+sum2

    return loss,gradient
