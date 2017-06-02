import IPython
import numpy as np
import pandas as pd

import yaml, io, os, requests, pickle, subprocess, csv

import matplotlib.pyplot as plt




divisions = 40
ymin = 1980
ymax = 2016

df_val = pd.read_csv("../usda_yield/county_yield_processed.csv")
df_val["filename"] = "county_level/" + df_val["name"].str.replace(", ", " ") + ".csv"

counties = df_val.filename.unique()

X_data = []
Y_data = []
county_data = []
year_data = []

for county_num, county in enumerate(counties):
	print (county)
	try:
		csv = file("extract/" + county)
	except:
		continue
	for i in range(0, 6): csv.readline()
	df = pd.read_csv(csv)

	for year in range(ymin, ymax+1):

		X_data_year = np.zeros((divisions, 7))

		for block in range(0, divisions):
			subsection = df[df.year == year][df.yday//(365//divisions)==block]
			vals = subsection.mean()

			X_data_year[block, :] = [vals['dayl (s)'], vals['prcp (mm/day)'], vals['srad (W/m^2)'], 
				vals['swe (kg/m^2)'], vals['tmax (deg c)'], vals['tmin (deg c)'], vals['vp (Pa)']]

		dd = df_val[df_val.filename==county]
		dd = dd[dd.year == year]

		if (len(dd['yield'])) == 0:
			continue

		X_data.append(X_data_year)
		Y_data.append(dd['yield'].mean())
		county_data.append(county)
		year_data.append(year)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

np.savez_compressed("features_raw.npz", X_data=X_data, Y_data=Y_data, county_data=county_data, year_data=year_data)

IPython.embed()
