import IPython
import numpy as np
import pandas as pd

import yaml, io, os, requests, pickle

import matplotlib.pyplot as plt

def find_nearest_counties(locations, position, n=3):
	dists = np.array([(x**2 + y**2)*1.0 for (x,y) in locations-position])
	idxs = dists.argsort()
	return idxs[0:n], dists[[idxs[0:n]]]

data_dict = np.load("data/data_cache.npz")

data = data_dict["data"]
target = data_dict["target"]
locations = data_dict["locations"]

#plt.scatter(locations[:, 0], locations[:, 1], c=target[:, 0]); plt.show()

locations[:, 0] -= np.min(locations[:, 0])
locations[:, 1] -= np.min(locations[:, 1])

X_MAX = np.max(locations[:, 0]) + 2
Y_MAX = np.max(locations[:, 1]) + 2

data_geofilled = np.zeros((X_MAX, Y_MAX,) + data.shape[1:])
target_geofilled = np.zeros((X_MAX, Y_MAX, target.shape[1]))

for i in range(0, X_MAX): 
	if i%10 == 0: print (i)
	for j in range(0, Y_MAX):
		counties, distances = find_nearest_counties(locations, (i, j), n=4)
		#print (counties, distances)
		distances += 0.0001 #Avoid div 0 errors

		weighted_sum_data = np.zeros(data.shape[1:])
		weighted_sum_target = np.zeros(target.shape[1])
		
		total = 0.0
		for county, dist in zip(counties, distances):
			weighted_sum_data += data[county, :]*1.0/(dist)
			weighted_sum_target += target[county, :]*1.0/(dist)
			total += 1.0/(dist)

		weighted_mean_data = weighted_sum_data/total
		weighted_mean_target = weighted_sum_target/total
		#print (i, j, weighted_mean, counties, distances)
		data_geofilled[i, j] = weighted_mean_data
		target_geofilled[i, j] = weighted_mean_target


plt.imshow(target_geofilled[:, :, 0]); plt.savefig("results/crop_yield_map.png")
plt.imshow(target_geofilled[:, :, 12]); plt.savefig("results/crop_yield_map_2.png")
plt.imshow(data_geofilled[:, :, 0, 3, 3]); plt.savefig("results/feature_map_1.png")

np.savez("data/data_cache_geofill.npz", data=data_geofilled, target=target_geofilled, locations=locations)