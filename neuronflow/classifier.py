import numpy as np
from utility import sigma

class logistic:
    def __init__(self,X,Y):
        if(X.ndim==1):
            self.X=X.reshape(X.shape[0],1)
        else: 
            self.X=X.T
        self.Y=Y.reshape(Y.shape[0],1)
        #print(self.X)
        #print(self.Y)
    def initialize_weights(self):
        n=self.X.shape[0]
        initial_w=np.zeros(n)
        return initial_w
    def compute_cost(self,w,b):
        m,n=self.X.shape



x=np.array([[1,4,6,7,8,10,-1],[10,20,30,4,5,6,7]])
y=np.array([0,0,0,1,1,1,0])
model=logistic(x,y)
model.initialize_weights()