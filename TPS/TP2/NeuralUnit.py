import numpy as np
from utils import *
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
        
    def forward(self,i):
        z = 0
        for k,prevNU in enumerate(self.preceding):
            z += self.w[k]*prevNU.forward(i)
        z += self.b
        z = sgm(z)
        self.z = z
        return z
        
    def backprop(self,i,delta):
        self.delta = np.zeros(self.w.shape)
        self.w_grad = np.zeros(self.w.shape)
        self.b_grad = 0
        for v in range(len(self.preceding)):
            self.delta[v] = delta[self.u]*self.z*(1-self.z)*self.w[v]
            self.w_grad[v] = self.z*(1-self.z)*self.preceding[v].z*delta[self.u]
        self.b_grad = self.z*(1-self.z)*delta[self.u]