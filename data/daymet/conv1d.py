
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

from keras.layers import Input, Reshape, Permute, Flatten, Dense, Lambda, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Cropping2D
from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D, ZeroPadding1D, SpatialDropout1D, Cropping1D, AveragePooling1D
from keras.models import Model, load_model
from keras.optimizers import SGD, adadelta
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import objectives, optimizers

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
X = X.reshape(data.shape)


Y = target.flatten()

#X, Y = shuffle(X, Y)

print (X.shape, Y.shape)

dropout = 0.43

input_img = Input((40, 7))
x = Convolution1D(10, 5, padding='same', activation='tanh')(input_img)
x = Convolution1D(15, 5, padding='same', activation='relu')(x)
x = MaxPooling1D(2) (x)
x = Dropout (rate=dropout) (x)
x = Convolution1D(20, 5, padding='same', activation='tanh') (x)
x = Convolution1D(30, 5, padding='same', activation='relu') (x)
x = MaxPooling1D(2) (x)
x = Dropout (rate=dropout) (x)
x = Convolution1D(40, 5, padding='same', activation='relu') (x)
x = Convolution1D(50, 5, padding='same', activation='relu') (x)
x = MaxPooling1D(2) (x)
x = Dropout (rate=dropout) (x)
x = Convolution1D(40, 4, padding='same', activation='relu') (x)
x = Convolution1D(20, 3, padding='same', activation='tanh') (x)
x = Flatten() (x)
x = Dense(200, activation='sigmoid') (x)
x = Dropout (rate=dropout) (x)
x = Dense(1, activation='linear') (x)

clf = Model(input_img, x)

#sgd = keras.optimizers.SGD(lr=1e-1, decay=1e-6, momentum=0.86, nesterov=True)

clf.compile(optimizer='adadelta', loss='mse')
clf.summary()

#clf = LinearRegression()
#clf = SVR()
#clf = BayesianRidge(compute_score=True)


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)


clf.fit(X_train, Y_train, epochs=100, batch_size=256,
	verbose=1, validation_data=(X_val, Y_val))

Y_pred = clf.predict(X_val)[:, 0]

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
