import numpy as np

class linear:
    def __init__(self,X,Y,max_itr=100000,lr=0.01,regularization=None,lambda1=1,lambda2=1):
        if(X.ndim==1):
            self.X=X.reshape(X.shape[0],1)
        else:
            self.X=X.T
        self.Y=Y
        self.max_itr=max_itr
        self.lr=lr
        self.regularization=regularization
        if(self.regularization=="elastic"):
            lambda1=0.5
            lambda2=0.5
        self.lambda1=lambda1
        self.lambda2=lambda2

    def initialize_weights(self):
        n=self.X.shape[1]
        initial_w=np.zeros(n)
        return initial_w
    
    def gradient_calc(self,w,b):
        f_wb=np.dot(self.X,w)+b
        j_wb=f_wb-self.Y
        lambda_val=[0,0]
        if(self.regularization=="l2"):
            lambda_val[1]=self.lambda2
        elif(self.regularization=="l1"):
            lambda_val[0]=self.lambda1
        elif(self.regularization=="elastic"):
            lambda_val=[self.lambda1,self.lambda2]
        dj_db=j_wb.sum()
        dj_db=dj_db/(len(self.Y))
        dj_dw=[]
        for i in range(len(w)):
            temp=j_wb*self.X[:,i]
            temp=temp.sum()
            temp=temp/(len(self.Y)) + (lambda_val[0]/(2*len(self.Y)))+ (lambda_val[1]*w[i])/len(self.Y)  #changed 
            dj_dw.append(float(temp))
        return dj_dw,dj_db
    def gradient_descent(self,w,b):
        threshold=0.000000000001
        db_temp=10000
        for i in range(self.max_itr):
            dw,db=self.gradient_calc(w,b)
            #print(dw)
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
    def fit(self):
        init_w=self.initialize_weights()
        init_b=0
        w,b=self.gradient_descent(init_w,init_b)
        return w,b
    def predict(self,x_test):
        if(x_test.ndim==1):
            x_test=x_test.reshape(x_test.shape[0],1)
        else: 
            x_test=x_test.T
        x=np.dot(x_test,self.w)+self.b
        return(x)
