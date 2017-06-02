
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import IPython


data_dict = np.load("county_level_features.npz")

X_data = data_dict["X_data"]
Y_data = data_dict["Y_data"]
county_data = data_dict["county_data"]
year_data = data_dict["year_data"]

df = pd.read_csv("../usda_yield/county_yield_processed.csv")
df = df[["name", "latitude", "longitude"]]
df = df.drop_duplicates()

df['filename'] = 'county_level/' + df["name"].str.replace(",", "") + ".csv"

county_dict = {}

for ind, val in df.iterrows():
	county_dict[val.filename] = (val.latitude, val.longitude)

locations = []

for county in county_data:
	locations.append(county_dict[county])
locations = np.array(locations)

np.savez_compressed("county_features.npz", X_data=X_data, Y_data=Y_data, 
	county_data=county_data, year_data=year_data, location_data=locations)

IPython.embed()