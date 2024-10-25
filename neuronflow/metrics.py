import numpy as np
import math

def mse(y,y_pred):
    y=y.reshape(y.shape[0],1)
    y_pred=y_pred.T
    squared_diff=(y-y_pred)**2
    diff_sum=squared_diff.sum()
    return diff_sum/y_pred.shape[0]

def rmse(y,y_pred):
    mse_score=mse(y,y_pred)
    return math.sqrt(mse)

def mae(y,y_pred):
    y=y.reshape(y.shape[0],1)
    y_pred=y_pred.T
    diff=abs(y_pred-y)
    return diff.sum()

def r2(y,y_pred):
    y=y.reshape(y.shape[0],1).T
    y_pred=y_pred.T
    diff=y-y_pred
    upper=(diff**2).sum()
    diff2=y-y.mean()
    lower=(diff2**2).sum()
    score=1-(upper/lower)
    return score