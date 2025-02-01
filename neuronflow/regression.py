import numpy as np

class linear:
    def __init__(self,X,Y,max_itr=100000,lr=0.01):
        if(X.ndim==1):
            self.X=X.reshape(X.shape[0],1)
        else:
            self.X=X.T
        self.Y=Y
        self.max_itr=max_itr
        self.lr=lr
    def initialize_weights(self):
        n=self.X.shape[1]
        initial_w=np.zeros(n)
        return initial_w
    def gradient_calc(self,w,b):
        f_wb=np.dot(self.X,w)+b
        j_wb=f_wb-self.Y
        lambda_val=[0,0]
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
    def predict(self,x):
        pass


model=linear(np.array([1,2,3,4]),np.array([2,4,6,8]))
vals=model.fit()
print(vals)