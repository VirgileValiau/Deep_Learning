class InputUnit:
    
    def __init__(self,data):
        self.data = data        # one column of matrix X
        self.n = data.shape[0]  # dataset size
        self.k = 0              # layer number
        self.z = 0              # unit output
        
    def plug(self,aUnit):
        aUnit.preceding.append(self)
        aUnit.npr += 1
        
    def forward(self,i):
        self.z = self.data[i]
        return self.z
        