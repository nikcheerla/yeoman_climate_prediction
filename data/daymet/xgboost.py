
import IPython

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, cross_val_predict, cross_val_score
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

import xgboost as xgb


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

clf = RandomForestRegressor(n_estimators=40, criterion='rmse')
#clf = LinearRegression()
#clf = SVR()
#clf = BayesianRidge(compute_score=True)

clf = xgb.XGBRegressor(max_depth=28,
           n_estimators=19000,
           min_child_weight=15,
           learning_rate=0.04,
           subsample=0.76,
           colsample_bytree=0.76,
           seed=4242)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)


clf.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], 
	verbose=True, eval_metric=['rmse'], early_stopping_rounds=50)

Y_pred = clf.predict(X_val, ntree_limit=clf.best_iteration)

r, p = pearsonr(Y_val, Y_pred)
print ("Correlation: ",  r)
print ("Variance explained: ",  r ** 2)
print ("P-value: ",  p ** 2)
print ("RMSE: ", mean_squared_error(Y_val, Y_pred)**0.5)

Y_pred = Y_pred[Y_val != 0]
Y_val = Y_val[Y_val != 0]

print ("MAPE: ", mean_absolute_percentage_error(Y_val, Y_pred))


target_pred = Y_pred.reshape(-1, 16)

for i in range(0, 16):
	print ("Correlation at year ", i + 1996, ": ", pearsonr(target[:, i], target_pred[:, i]))

plt.scatter(locations[:, 0], locations[:, 1], c=(target_pred[:, 0] - target[:, 0])**2); plt.show()
