from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from xgboost import XGBoostScratch
import numpy as np
import matplotlib.pyplot as plt

X,y = make_regression(
    n_samples=200,
    n_features=10,
    noise=10,
    random_state=42
)
model = XGBoostScratch(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    lam = 1.0,
    gamma=0.0,
    colsample=0.8,
    loss="mse"
)
model.fit(X,y)
preds = model.predict(X)

mse = mean_squared_error(y,preds)
print("Final Training MSE",mse)

plt.plot(model.train_loss,label = "Train Loss")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.show()