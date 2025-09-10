import numpy as np

class MSELoss:
    def loss(self,y,y_pred):
        return 0.5(np.mean((y - y_pred)**2))
    def grad(self,y,y_pred):
        return (y_pred - y)
    def hess(self,y,y_pred):
        return np.ones_like(y)
    