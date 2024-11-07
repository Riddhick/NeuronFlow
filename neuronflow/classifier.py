import numpy as np
from utility import sigma

class logistic:
    def __init__(self,X,Y,epoch):
        if(X.ndim==1):
            self.X=X.reshape(X.shape[0],1)
        else: 
            self.X=X.T
        self.Y=Y
        self.epoch=epoch
        #print(self.X)
        #print(self.Y)
    def initialize_weights(self):
        n=self.X.shape[1]
        initial_w=np.zeros(n)
        return initial_w
    def compute_cost(self,w,b):
        m,n=self.X.shape
    def gradient_calculate(self,w,b):
        z=np.dot(self.X,w)+b
        f_wb=sigma(z)
        j_wb=f_wb-self.Y
        dj_db=j_wb.sum()
        dj_db=dj_db/(len(self.Y))
        dj_dw=[]
        for i in range(len(w)):
            temp=j_wb*self.X[:,i]
            temp=temp.sum()
            temp=temp/(len(self.Y))   
            dj_dw.append(float(temp))
        #print(dj_dw)
        return dj_dw,dj_db
    def gradient_descent(self,w,b):
        for i in range(self.epoch):
            dw,db=self.gradient_calculate(w,b)
            for i in range(len(dw)):
                dw[i]=dw[i]*0.1
            w=w-dw
            b=b-0.1*db
        return w,b
    def fit(self):
        init_w=self.initialize_weights()
        init_b=0
        w,b=self.gradient_descent(init_w,init_b)
        print(w,b)



x=np.array([[0.1,1.2,1.5,2.0,1.0,2.5],[1.1,.9,1.5,1.8,2.5,.5]])
y=np.array([0,0,1,1,1,0])
model=logistic(x,y,10)
#model.initialize_weights()
model.fit()