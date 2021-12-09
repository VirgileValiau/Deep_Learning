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
                layer.append(Nu)  
            net.append(layer) 
        net.append([Loss(y,len(archi))])
        self.net = net
        
        # plug network
        for l in range(len(net)-1):
            for n in range(len(net[l])):
                for next_neuron in net[l+1]:
                    net[l][n].plug(next_neuron)   
                    if l!=0 and l != len(net)-1:
                        net[l][n].reset_params()

    def forward(self,i):
        return self.net[-1][-1].forward(i)
    
    def backprop(self,i):
        for k in range(self.K,0,-1):
            if k == self.K:
                self.net[k][0].backprop(i)
                deltas = self.net[k][0].delta
            else:
                deltas_new = np.zeros((self.net[k][0].npr,))
                for u in range(len(self.net[k])):
                    self.net[k][u].backprop(i,deltas)
                    deltas_new += self.net[k][u].delta
                deltas = deltas_new
                
                
    def update(self,eta):
        for l in range(1,len(self.net)-1):
            for u in self.net[l]:
                u.w -= eta*u.w_grad
                u.b -= eta*u.b_grad
                
    def train(self,epochs,eta):
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("epoch ",epoch,"/",epochs,'...')
            for i in range(self.n):
                self.forward(i)
                self.backprop(i)
                self.update(eta)
        print("epoch",epochs,"/",epochs,"...")
                
    def predict(self,i):
        return self.net[-2][0].forward(i)
