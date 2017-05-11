import IPython
import numpy as np
import pandas as pd

import yaml, io, os, requests, pickle

import matplotlib.pyplot as plt

def unique_rows(data):
	uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
	return uniq.view(data.dtype).reshape(-1, data.shape[1])



#FIPS countries: Illinois, Indiana, Iowa

df = pd.read_csv("data/maize.alldat.nolatlon.csv")
fips_lookup = pd.read_csv("data/fips_lookup.csv")


county_location = {}
location_cache = "data/county_location.pkl"

if os.path.exists(location_cache):
	print ("Loading: ", location_cache)
	with open(location_cache, 'r') as stream:
		county_location = pickle.load(stream)

else:

	for i in range(0, len(fips_lookup)):
		code = fips_lookup['statecode'][i]*1000 + fips_lookup['countycode'][i]
		address = fips_lookup['countyname'][i] + " " + fips_lookup['state'][i]
		print (code, address)
		address = address.replace(' ', '+')
		rest_api_address = "http://www.datasciencetoolkit.org/street2coordinates/" + address
		request = requests.get(rest_api_address).json()
		request = request[request.keys()[0]]
		longitude, latitude = request['longitude'], request['latitude']
		county_location[code] = (longitude, latitude)

	with open(location_cache, 'w') as outfile:
		pickle.dump(county_location, outfile)

MIN_YEAR = 1996
MAX_YEAR = 2011
SCALE = 30

data = []
target = []
locations = []

county_at = {}

for county in df['FIPS'].unique():
	
	latitude, longitude = 0, 0
	if county in county_location:
		latitude, longitude = county_location[county]
	else:
		lat1, long1 = county_location[county-1]
		lat2, long2 = county_location[county+1]
		latitude, longitude = (lat1 + lat2)/2, (long1 + long2)/2

	longitude = int((longitude)*SCALE)
	latitude = int((latitude)*SCALE)

	if (longitude, latitude) not in county_at:
		county_at[(longitude, latitude)] = county
		print (longitude, latitude)
	else:		
		df.loc[(df.FIPS == county), 'FIPS'] = county_at[(longitude, latitude)]

print len(df['FIPS'].unique())


for county in df['FIPS'].unique():

	min_year = df[(df.FIPS == county)].YEAR.min()
	max_year = df[(df.FIPS == county)].YEAR.max()
	
	if min_year > MIN_YEAR or max_year < MAX_YEAR: continue

	drop_data = False

	county_year_data = np.zeros((MAX_YEAR - MIN_YEAR + 1, 4, 5))
	county_year_target = np.zeros((MAX_YEAR - MIN_YEAR + 1))

	for year in range(MIN_YEAR, MAX_YEAR+1):
		subdf = df[(df.FIPS == county)][(df.YEAR == year)]
		mean = subdf.mean()
		i = year - MIN_YEAR
		county_year_data[i, 0, 0] = mean["Tmin_.30_0"]
		county_year_data[i, 0, 1] = mean["Tmin_1_30"]
		county_year_data[i, 0, 2] = mean["Tmin_31_60"]
		county_year_data[i, 0, 3] = mean["Tmin_61_90"]
		county_year_data[i, 0, 4] = mean["Tmin_91_120"]
		county_year_data[i, 1, 0] = mean["Tmax_.30_0"]
		county_year_data[i, 1, 1] = mean["Tmax_1_30"]
		county_year_data[i, 1, 2] = mean["Tmax_31_60"]
		county_year_data[i, 1, 3] = mean["Tmax_61_90"]
		county_year_data[i, 1, 4] = mean["Tmax_91_120"]
		county_year_data[i, 2, 0] = mean["PRCP_.30_0"]
		county_year_data[i, 2, 1] = mean["PRCP_1_30"]
		county_year_data[i, 2, 2] = mean["PRCP_31_60"]
		county_year_data[i, 2, 3] = mean["PRCP_61_90"]
		county_year_data[i, 2, 4] = mean["PRCP_91_120"]
		county_year_data[i, 3, 0] = mean["VPD_.30_0"]
		county_year_data[i, 3, 1] = mean["VPD_1_30"]
		county_year_data[i, 3, 2] = mean["VPD_31_60"]
		county_year_data[i, 3, 3] = mean["VPD_61_90"]
		county_year_data[i, 3, 4] = mean["VPD_91_120"]
		county_year_target[i] = mean["TRUEYIELD"]

		if np.isnan(mean['TRUEYIELD']):
			drop_data = True
			continue

	if drop_data: continue

	data.append(county_year_data)
	target.append(county_year_target)

	latitude, longitude = 0, 0
	if county in county_location:
		latitude, longitude = county_location[county]
	else:
		lat1, long1 = county_location[county-1]
		lat2, long2 = county_location[county+1]
		print (lat1, long1, lat2, long2)
		latitude, longitude = (lat1 + lat2)/2, (long1 + long2)/2

	longitude = int((longitude)*SCALE)
	latitude = int((latitude)*SCALE)

	print (longitude, latitude)

	locations.append((latitude, longitude))

data = np.array(data)
target = np.array(target)
locations = np.array(locations)

print (len(locations))
print (len(unique_rows(locations)))

#plt.scatter(locations[:, 0], locations[:, 1]); plt.show()

np.savez("data/data_cache.npz", data=data, target=target, locations=locations)