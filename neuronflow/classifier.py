import numpy as np
from utility import sigma

class logistic:
    def __init__(self,X,Y,max_iter=10000,lr=0.01,activation="sigmoid",regularization=None):
        if(X.ndim==1):
            self.X=X.reshape(X.shape[0],1)
        else: 
            self.X=X.T
        self.Y=Y
        self.max_iter=max_iter
        self.lr=lr
        self.activation="sigmoid"
        self.regularization=None
        #print(self.X)
        #print(self.Y)

    def initialize_weights(self):
        n=self.X.shape[1]
        initial_w=np.zeros(n)
        return initial_w
    
    def gradient_calculate(self,w,b):
        z=np.dot(self.X,w)+b
        if(self.activation=="sigmoid"):
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
        threshold=0.0000001
        db_temp=10000
        for i in range(self.max_iter):           
            dw,db=self.gradient_calculate(w,b)
            if(abs(db-db_temp)<threshold):
               break
            for i in range(len(dw)):
                dw[i]=dw[i]*self.lr
            w=w-dw
            b=b-self.lr*db
            db_temp=db 
        self.w=w
        self.b=b
        return w,b
    
    def predict(self,x_test):
        if(x_test.ndim==1):
            x_test=x_test.reshape(x_test.shape[0],1)
        else: 
            x_test=x_test.T
        x=np.dot(x_test,self.w)+self.b
        if(self.activation=="sigmoid"):
            f_wb=sigma(x)
        for i in range(len(f_wb)):
            if(f_wb[i]>0.5):
                f_wb[i]=1
            else:
                f_wb[i]=0
        print(f_wb)

    def fit(self):
        init_w=self.initialize_weights()
        init_b=0
        w,b=self.gradient_descent(init_w,init_b)
        return w,b



