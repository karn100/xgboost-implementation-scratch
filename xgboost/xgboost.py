import numpy as np
from .losses import MSELoss,LogisticLoss
from .tree import XGBoostTree

class XGBoostScratch:
    def __init__(
            self,
            n_estimators = 100,
            learning_rate = .1,
            max_depth = 3,
            min_sample_split = 2,
            lam = 1.0,
            gamma = 0.0,
            colsample = 1,
            loss = "mse"
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lam = lam
        self.gamma = gamma
        self.colsample = colsample
        self.loss_name = loss

        self.trees = []
        self.loss_fn = MSELoss() if loss == "mse" else LogisticLoss()
        self.initial_pred = None
        self.train_loss = []
    
    def fit(self,X,y):
        n_samples,n_features = X.shape

        if self.loss_name == "mse":
            self.initial_pred = np.mean(y)
        else:
            p = np.mean(y)
            self.initial_pred = np.log(p/(1 - p + 1e-15))
        
        y_pred = np.full(n_samples,self.initial_pred)

        for m in range(self.n_estimators):

            grad = self.loss_fn.grad(y,y_pred)
            hess = self.loss_fn.hess(y,y_pred)    

            col_idx = np.random.choice(n_features,max(1,int(n_features*self.colsample)),replace=False)
            X_sub = X[:,col_idx]

            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                lam = self.lam,
                gamma=self.gamma,
                feature_indices=col_idx
            )
            tree.fit(X_sub,grad,hess)
            tree.feature_indices = col_idx
            update = tree.predict(X_sub)

            y_pred += self.learning_rate*update
            self.trees.append(tree)
            self.train_loss.append(self.loss_fn.loss(y,y_pred))
            print(f"Iteration: {m+1}/{self.n_estimators}, Loss: {self.train_loss[-1]:.5f}")
    
    def predict(self,X):
        n_samples = X.shape[0]

        y_pred = np.full(n_samples,self.initial_pred)

        for tree in self.trees:
            X_sub = X[:,tree.feature_indices]
            y_pred += self.learning_rate*tree.predict(X_sub)

        return self.loss_fn.transform(y_pred)
    
        
