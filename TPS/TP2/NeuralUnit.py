import numpy as np

class NeuralUnit:
    
    def __init__(self,k,u):
        self.u = u          # unit number
        self.preceding = [] # list of preceding neurons
        self.npr = 0        # length of list preceding
        self.following = [] # list of following neurons
        self.nfo = 0        # length of list following
        self.k = k          # layer number
        self.w = 0          # unit weight
        self.b = 0          # unit intercept
        self.z = 0          # unit output
        
    def reset_params(self):
        self.w = np.random.randn(self.npr)
        self.b = np.random.randn()
    
        
    def plug(self,aUnit):
        self.following.append(aUnit)
        self.nfo += 1
        aUnit.preceding.append(self)
        aUnit.npr += 1
        