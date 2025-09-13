import numpy as np

class MSELoss:
    def loss(self,y,y_pred):
        return 0.5*(np.mean((y - y_pred)**2))
    
    def grad(self,y,y_pred):
        return (y_pred - y)
    
    def hess(self,y,y_pred):
        return np.ones_like(y)
    
    def transform(self,y_pred):
        return y_pred

class LogisticLoss:

    def __init__(self):
        self.eps = 1e-15

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))   
     
    def loss(self,y,y_pred):
        p = self.sigmoid(y_pred)
        p = np.clip(p,self.eps,1-self.eps)
        return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))  #negative sign is used here because p range is 0 to 1 and log values are negative for this range , so loss cant be negative 
    
    def grad(self,y,y_pred):
        p = self.sigmoid(y_pred)
        return p - y
    
    def hess(self,y,y_pred):
        p = self.sigmoid(y_pred)
        return p*(1 - p)
    
    def transform(self,y_pred):
        return self.sigmoid(y_pred)
    