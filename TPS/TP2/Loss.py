class Loss:
    
    def __init__(self,y,k):
        self.preceding = [] # list of preceding neurons
        self.npr = 0        # length of list preceding
        self.y = y          # array of class labels of the training data
        self.k = k          # layer index