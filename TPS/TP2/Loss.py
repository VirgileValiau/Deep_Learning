from utils import *

class Loss:
    
    def __init__(self,y,k):
        self.preceding = [] # list of preceding neurons
        self.npr = 0        # length of list preceding
        self.y = y          # array of class labels of the training data
        self.k = k          # layer index
        
    def forward(self,i):
        label = onehot(self.y,i)
        zin = self.preceding[0].forward(i)
        if self.y[i] == 0:
            return -np.log(1-zin)
        else:
            return -np.log(zin)
        
    