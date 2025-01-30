import numpy as np

class linear:
    def __init__(self,X,Y,max_iter=1000,lr=0.01):
        self.X=X.reshape(X.shape[0],1)
        self.Y=Y.reshape(Y.shape[0],1)
        self.max_iter=max_iter
        self.lr=lr

    def fit_gradient(self):
        print(self.X)
        init_w=0
        init_b=0
        #dw,db=self.gradient_descent(init_w,init_b)
        threshold=0.000001
        for i in range(100000):
            dw,db=self.gradient_descent(init_w,init_b)
            init_w=init_w-self.lr*dw
            init_b=init_b-self.lr*db
        print(init_b,init_w)

    def gradient_descent(self,w,b):
        #threshold=0.000001
        z=np.dot(self.X,w)+b
        error=z-self.Y
        dw=error*self.X
        dw=np.sum(dw)/self.X.shape[0]
        db=np.sum(error)/self.X.shape[0]
        return dw,db
    
    def fit(self):
        ones=np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
        self.X=np.append(ones,self.X,axis=1)
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
        return values
    


class multilinear(linear):
    def __init__(self,X,Y):
        self.X=X.T
        #ones=np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
        #self.X=np.append(ones,self.X,axis=1)
        self.Y=Y.reshape(Y.shape[0],1)
        
    def fit(self):
        thetas=super().fit()
        self.thetas=thetas
        return self.thetas
    
    def value(self,x):
        x=x.T
        ones=np.ones(x.shape[0]).reshape(x.shape[0],1)
        x=np.append(ones,x,axis=1)
        values=np.matmul(x,self.thetas)
        return values
    

## have to change the fit
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
        return values
        

#test case 
x=np.array([1,2,3,4,5])
y=np.array([6,8,10,12,14])
model=linear(x,y)
model.fit_gradient()