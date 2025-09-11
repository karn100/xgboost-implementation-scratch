import numpy as np

class BoostingTreeNode:
    def __init__(self,grad,hess,lam,depth = 0,max_depth = 3,gamma = 0.0):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None

        self.grad = grad
        self.hess = hess
        self.lam = lam
        self.depth = depth
        self.gamma = gamma
        self.max_depth = max_depth
    
    def calc_leaf_value(self):
        G = np.sum(self.grad)
        H = np.sum(self.hess)
        return -G/(H + self.lam)
    
    def calc_leaf_gain(self,G,H,GL,GR,HL,HR):
        gain = 0.5*((GL**2/(HL + self.lam)) + (GR**2/(HR + self.lam)) - (G/H + self.lam)) - self.gamma
        return gain
    
    def best_split(self,X):
        n_samples,n_features = X.shape
        G,H = np.sum(self.grad),np.sum(self.hess)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                left_idx = X[:,feature] <= threshold
                right_idx =~ left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                GL,HL = np.sum(self.grad[left_idx]), np.sum(self.hess[left_idx])
                GR,HR = np.sum(self.grad[right_idx]), np.sum(self.hess[right_idx])

                gain = self.calc_leaf_gain(G,H,GL,GR,HL,HR)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature,best_threshold,best_gain
    
    def build(self,X):
        if self.depth >= self.max_depth:
            self.value = True
            self.value = self.calc_leaf_value()
            return
        best_feature,best_threshold,best_gain = self.best_split(X)

        if best_gain <= 0 or best_feature == 0:
            self.value = True
            self.value = self.calc_leaf_value()
            return
        
        left_idx = X[:,best_feature] <= best_threshold
        right_idx =~ left_idx

        self.feature_index = best_feature
        self.threshold = best_threshold

        self.left = BoostingTreeNode(
            grad = self.grad[left_idx],
            hess = self.hess[left_idx],
            lam = self.lam,
            depth = self.depth + 1,
            max_depth=self.max_depth,
            gamma= self.gamma
        )
        self.left.build(X[left_idx])

        self.right = BoostingTreeNode(
             grad = self.grad[right_idx],
            hess = self.hess[right_idx],
            lam = self.lam,
            depth = self.depth + 1,
            max_depth=self.max_depth,
            gamma= self.gamma
        )
        self.right.build(X[right_idx])
    
    def pred_row(self,x):
        if self.is_leaf:
            return self.value
        if x[self.feature_index] <= self.threshold:
            return self.left.pred_row(x)
        else:
            return self.right.pred_row(x)
        
class XGBoostTree:
    def __init__(self,max_depth = 3,min_sample_split = 2,lam = 1.0,gamma = 0.0,feature_indices = None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lam = lam
        self.gamma = gamma
        self.feature_indices = feature_indices
        self.root = None
    
    def fit(self,X,grad,hess):

        self.root = BoostingTreeNode(
            grad=grad,
            hess=hess,
            lam = self.lam,
            depth = 0,
            max_depth= self.max_depth,
            gamma= self.gamma
        )
        self.root.build(X)
    
    def predict(self,X):
        preds = np.array([self.root.pred_row(x) for x in X])
        return preds
    