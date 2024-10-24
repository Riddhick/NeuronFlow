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
        if(np.linalg.det(X_t_X)==0):
            X_inv=np.linalg.pinv(X_t_X)
        else:
            X_inv=np.linalg.inv(X_t_X)
        temp=np.matmul(X_inv,xTranspose)
        thetas=np.matmul(temp,self.Y)
        self.thetas=thetas
        return self.thetas
    
    def value(self,x):
        x=x.reshape(x.shape[0],1)
        ones=np.ones(x.shape[0]).reshape(x.shape[0],1)
        x=np.append(ones,x,axis=1)
        values=np.matmul(x,self.thetas)
        return values.T
    
    def r_square(self,x,y):
        y=y.reshape(y.shape[0],1).T
        y_pred=self.value(x)
        diff=y-y_pred
        upper=(diff**2).sum()
        diff2=y-y.mean()
        lower=(diff2**2).sum()
        score=1-(upper/lower)
        return score


class multilinear(linear):
    def __init__(self,X,Y):
        self.X=X.T
        ones=np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
        self.X=np.append(ones,self.X,axis=1)
        self.Y=Y.reshape(Y.shape[0],1)
        
    def fit(self):
        thetas=super().fit()
        self.thetas=thetas
        return self.thetas
    
    def value(self,x):
        ones=np.ones(x.shape[0]).reshape(x.shape[0],1)
        x=np.append(ones,x,axis=1)
        values=np.matmul(x,self.thetas)
        return values.T
    
    def r_square(self,x,y):
        y=y.reshape(y.shape[0],1).T
        y_pred=self.value(x.T)
        diff=y-y_pred
        upper=(diff**2).sum()
        diff2=y-y.mean()
        lower=(diff2**2).sum()
        score=1-(upper/lower)
        return score


class polynomial(linear):
    def __init__(self,X,Y,degree):
        self.X=X.reshape(X.shape[0],1)
        ones=np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
        self.X=np.append(ones,self.X,axis=1)
        self.X=np.repeat(self.X,[1,degree],axis=1)
        self.Y=Y.reshape(Y.shape[0],1)
        self.X=self.X**np.arange(0,self.X.shape[1])
       
    def fit(self):
        thetas=super().fit()
        self.thetas=thetas
        return self.thetas
    
    def value(self,x):
        degree=self.thetas.shape[0]-1
        x=x.reshape(x.shape[0],1)
        ones=np.ones(x.shape[0]).reshape(x.shape[0],1)
        x=np.append(ones,x,axis=1)
        x=np.repeat(x,[1,degree],axis=1)
        x=x**np.arange(0,x.shape[1])
        values=np.matmul(x,self.thetas)
        return values.T
    
    def r_square(self, x, y):
        return super().r_square(x, y)
        

