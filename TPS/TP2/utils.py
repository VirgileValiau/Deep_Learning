import numpy as np

def sgm(x):
    return 1/(1+np.exp(-x))

def onehot(y,i):
    return 1-np.array(list(bin(y[i]+1)[2:].zfill(2))).astype(int)
