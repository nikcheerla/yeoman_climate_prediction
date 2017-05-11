
import IPython

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.svm import SVR
from sklearn.utils import shuffle
from scipy.stats import pearsonr

data_dict = np.load("data/data_cache.npz")

data = data_dict["data"]
target = data_dict["target"]
locations = data_dict["locations"]

X = data.reshape((-1, 20))
Y = target.flatten()

#X, Y = shuffle(X, Y)

print (X.shape, Y.shape)

clf = RandomForestRegressor(n_estimators=10, criterion='mse')
#clf = LinearRegression()
#clf = SVR(C=1.0)

Y_pred = cross_val_predict(clf, X, Y, cv=10)

r, p = pearsonr(Y, Y_pred)
print ("Correlation: ",  r)
print ("Variance explained: ",  r ** 2)
print ("P-value: ",  p ** 2)


target_pred = Y_pred.reshape(120, 16)

for i in range(0, 16):
	print ("Correlation at year ", i + 1996, ": ", pearsonr(target[:, i], target_pred[:, i]))

plt.scatter(locations[:, 0], locations[:, 1], c=(target_pred[:, 0] - target[:, 0])**2); plt.show()

IPython.embed()