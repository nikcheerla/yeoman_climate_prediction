
#code for data generation from heatmaps

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import glob, sys, os, random, time, logging, threading, subprocess
import scipy.io, scipy.misc

from sklearn.cross_validation import train_test_split
import progressbar
from keras.metrics import binary_crossentropy, binary_accuracy

from utils import evaluate_model_on_directory

import IPython

def totuple(a):
	try:
		return tuple(totuple(i) for i in a)
	except TypeError:
		return a


def tolist(a):
	try:
		return list(totuple(i) for i in a)
	except TypeError:
		return a

def normalize(arr):
	return (arr - arr.mean())/arr.std()





class AbstractGenerator(object):
	def __init__(self, samples_per_epoch=500, val_samples=100, batch_size=50, verbose=True):
		self.samples_per_epoch = samples_per_epoch
		self.val_samples = val_samples
		self.batch_size = batch_size
		self.verbose = verbose

		self.load()

	def load(self):
		raise NotImplementedError()

	def checkpoint(self):
		#cmd = ['python', 'evaluate.py', '-w', self.model.checkpoint, '-f', 
		#	'datasets/icpr/train/A01_03_image.jpg', '-o', 'datasets/icpr/train/A01_03_pred.jpg']
		print "Checkpointing"

	def data(self, mode='train'):
		num_samples = self.samples_per_epoch if mode == 'train' else self.val_samples

		generator_embed = self.gen_sample_pair(mode=mode, num_samples=num_samples)

		window, target = next(generator_embed)
		window, target = np.array(window), np.array(target)
		batch_data = np.zeros((num_samples, ) + window.shape, dtype=window.dtype)
		batch_target = np.zeros((num_samples, ) + target.shape, dtype=target.dtype)

		for i in range(0, num_samples):
			window, target = next(generator_embed)
			batch_data[i] = window
			batch_target[i] = target

		generator_embed.close()

		return batch_data, batch_target




class GeofillGenerator(AbstractGenerator):
	def __init__(self, *args, **kwargs):
		self.features = kwargs.pop('features')
		self.target = kwargs.pop('target')
		self.years = kwargs.pop('years')
		self.locations = kwargs.pop('locations')
		self.window_size = kwargs.pop('window_size', 224)
		self.resolution = kwargs.pop('resolution', 40)
		self.k = kwargs.pop('k', 4)

		super(GeofillGenerator, self).__init__(*args, **kwargs)

	def load(self):
		self.locations = (np.array(self.locations)*self.resolution).astype(int)

	def gen_sample_pair(self, mode='train', num_samples=100):

		while True:
			def find_nearest_counties(locations, position, k=3):
				dists = np.array([(x**2 + y**2)*1.0 for (x,y) in locations-position])
				idxs = dists.argsort()
				return idxs[0:k], dists[[idxs[0:k]]]

			x = random.randint(self.locations[:, 0].min(), self.locations[:, 0].max())
			y = random.randint(self.locations[:, 1].min(), self.locations[:, 1].max())

			print (x, y)
			year = random.randint(min(self.years), max(self.years))
			current_year_mask = self.years == year

			xs, ys = x - self.window_size//2, y - self.window_size//2

			input_window = np.zeros((self.window_size, self.window_size) + self.features.shape[1:])

			for xc in range(xs, xs + self.window_size):
				for yc in range(ys, ys + self.window_size):
					
					counties, distances = find_nearest_counties(self.locations[current_year_mask], (xc, yc), k=self.k)
					#print (counties, distances)
					distances += 0.000001 #Avoid div 0 errors
					
					total = 0.0
					for county, dist in zip(counties, distances):
						input_window[xc - xs, yc - ys] += self.features[current_year_mask][county, :]*1.0/(dist)
						total += 1.0/(dist)

					input_window[xc - xs, yc - ys] /= total

			counties, distances = find_nearest_counties(self.locations, (x, y), k=self.k)
			#print (counties, distances)
			distances += 0.000001 #Avoid div 0 errors
			
			target = 0.0
			total = 0.0
			for county, dist in zip(counties, distances):
				target += self.target[county]*1.0/(dist)
				total += 1.0/(dist)

			target /= total

			yield input_window, target





if __name__ == "__main__":
	print ("Hi")



		