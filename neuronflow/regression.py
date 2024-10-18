import numpy as np

class linear:
    def __init__(self,X,Y):
        self.X=X.reshape(X.shape[0],1)
        self.Y=Y.reshape(Y.shape[0],1)
        ones=np.ones(X.shape[0]).reshape(X.shape[0],1)
        self.X=np.append(ones,self.X,axis=1)
    def fit(self):
        xTranspose=self.X.T
        X_t_X=np.matmul(xTranspose,self.X)
        X_inv=np.linalg.inv(X_t_X)
        temp=np.matmul(X_inv,xTranspose)
        thetas=np.matmul(temp,self.Y)
        self.thetas=thetas
    def value(self,x):
        x=x.reshape(x.shape[0],1)
        ones=np.ones(x.shape[0]).reshape(x.shape[0],1)
        x=np.append(ones,x,axis=1)
        values=np.matmul(x,self.thetas)
        return values.T
    

model1=linear(np.array([1,2,3,4]),np.array([3,5,7,9]))
model1.fit()
values=model1.value(np.array([5]))
print(values)