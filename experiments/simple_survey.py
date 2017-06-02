
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from generators import GeofillGenerator

import IPython


data_dict = np.load("data/daymet/county_features.npz")

data = data_dict["X_data"]
target = data_dict["Y_data"]
locations = data_dict["location_data"]
years = data_dict["year_data"]

datagen = GeofillGenerator(samples_per_epoch=4, val_samples=1, batch_size=50,
	features=data, target=target, locations=locations, years=years,
	window_size=80, resolution=10, k=4)

X = datagen.data(mode='train')
IPython.embed()