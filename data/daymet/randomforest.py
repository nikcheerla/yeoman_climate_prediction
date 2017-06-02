
import IPython

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


from scipy.stats import pearsonr


def mean_absolute_percentage_error(y_true, y_pred): 

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

data_dict = np.load("county_level_features.npz")

data = data_dict["X_data"]
target = data_dict["Y_data"]

X = data.reshape((len(data), -1))
X = preprocessing.scale(X)

Y = target.flatten()

#X, Y = shuffle(X, Y)

print (X.shape, Y.shape)

clf = RandomForestRegressor(n_estimators=50, criterion='mse')
#clf = LinearRegression()
#clf = SVR()
#clf = BayesianRidge(compute_score=True)

Y_pred = cross_val_predict(clf, X, Y, cv=10)

r, p = pearsonr(Y, Y_pred)
print ("Correlation: ",  r)
print ("Variance explained: ",  r ** 2)
print ("P-value: ",  p ** 2)
print ("RMSE: ", mean_squared_error(Y, Y_pred)**0.5)

Y_pred = Y_pred[Y != 0]
Y = Y[Y != 0]

print ("MAPE: ", mean_absolute_percentage_error(Y, Y_pred))


target_pred = Y_pred.reshape(-1, 16)

for i in range(0, 16):
	print ("Correlation at year ", i + 1996, ": ", pearsonr(target[:, i], target_pred[:, i]))

plt.scatter(locations[:, 0], locations[:, 1], c=(target_pred[:, 0] - target[:, 0])**2); plt.show()
