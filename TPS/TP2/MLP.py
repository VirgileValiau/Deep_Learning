from NeuralUnit import *
from InputUnit import *
from Loss import *

class MLP:
    
    def __init__(self,X,y,archi):
        self.archi = archi
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.K = len(archi)     # number of layers (including input layer)
        
        # creating network
        net = []
        for i in range(len(archi)):
            layer=[]
            for j in range(archi[i]):
                if i==0:
                    Nu = InputUnit(X[:,j])
                else :
                    Nu = NeuralUnit(i,j)
                    Nu.reset_params()
                layer.append(Nu)  
            net.append(layer) 
        net.append([Loss(y,len(archi))])
        self.net = net
        
        # plug network
        for l in range(len(net)-1):
            for n in range(len(net[l])):
                for next_neuron in net[l+1]:
                    net[l][n].plug(next_neuron)   
