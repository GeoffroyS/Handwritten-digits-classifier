#!/usr/bin/env python

import os
import struct

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import neuralnet

def load_mnist (path, kind='train'):
	"""
	load ubyte files downloaded from Yann LeCun's page
	http://yann.lecun.com/exdb/mnist/
	return a dictionary containing the 3 pd.DataFrames:
	train, validation, test
	"""

	df_dict = {}

	for kind in ['train', 'test']:
		labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
		pixels_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)

		# read 'magic' number (description of the file protocol) 
		# and number of lines 'n'
		with open(labels_path, 'rb') as labels_h:
			magic, n = struct.unpack('>II', labels_h.read(8))
			labels = np.fromfile(labels_h, dtype=np.uint8)

		with open(pixels_path, 'rb') as pixels_h:
			magic, num, rows, cols = struct.unpack('>IIII', pixels_h.read(16))
			pixels = np.fromfile(pixels_h, dtype=np.uint8).reshape(len(labels), 784)

		df = pd.DataFrame(pixels)
		df['target'] = labels

		if kind == 'train':
			X = df.iloc[:,:-1]
			y = df['target']
			X_train, X_validation, y_train, y_validation = train_test_split(
				X, 
				y, 
				train_size=50000, 
				test_size=10000, 
				random_state=0
				)
			X_train = X_train.applymap(_normalize_features)
			X_validation = X_validation.applymap(_normalize_features)
			#print(X_train.loc[0].tolist())
			df_train = pd.concat([X_train, y_train], axis=1, sort=False)
			df_validation = pd.concat([X_validation, y_validation], axis=1, sort=False)
			df_dict["train"] = df_train
			df_dict["validation"] = df_validation
		else:
			df.loc[:, df.columns != 'target'].applymap(_normalize_features)
			df_dict["test"] = df 

	return df_dict

def _normalize_features(value):
	return value/255

def _display_digits(datasets_dict, plot_type='zero_nine', digit=3):
	"""
	- plot_type = 'zero_nine':
	  take the first instances of the 10 digits
	  from the training set and display them
	- plot_type = 'same_digit':
	  display the first 10 instances of a single digit
	  ('digit' arg, 3 by default)
	"""
	fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
	ax = ax.flatten()
	X_train = datasets_dict["train"].iloc[:,0:784].to_numpy()
	y_train = datasets_dict["train"].iloc[:,-1].to_numpy()
	df_train = datasets_dict["train"]

	if plot_type == 'zero_nine':
		for i in range(10):
			img = X_train[y_train == i][0].reshape(28, 28)
			ax[i].imshow(img, cmap='Greys', interpolation='nearest')
	elif plot_type == 'same_digit':
		for i in range(10):
			img = X_train[y_train == digit][i].reshape(28, 28)
			ax[i].imshow(img, cmap='Greys', interpolation='nearest')

	ax[0].set_xticks([])
	ax[0].set_yticks([])
	plt.tight_layout()
	plt.show()

def _display_digits_distrib(datasets_dict):
	"""
	Plot the distribution of the 10 digits in the training dataset
	using a simple bar chart
	"""
	df_train = datasets_dict["train"]
	digits_dict = {}
	for i in range(10):
		digits_dict[i] = df_train.loc[df_train['target'] == i].shape[0]
		print('number of instances of {} in dataset: {}'
			.format(i, digits_dict[i]))

	x = [key for key in digits_dict]
	y = [val for val in digits_dict.values()]

	plt.title('Digits distribution')
	plt.xlabel('Digits')
	plt.ylabel('Number of instances')
	plt.bar(x, y)
	plt.show()

def _df_to_ndarray(data_df, dataset_type):
	"""
	This returns a list containing x tuples with x=df.shape[0]
	Each tuple contains either
	- 2 ndarrays of shapes (784,1) and (10,1) (for the training set)
	- 1 ndarray of shape (784,1) and an int corresponding to the target (for the validation and test sets)
	"""
	data = data_df.loc[:, data_df.columns != 'target'].apply(lambda row: row.to_numpy(dtype='float32').reshape(784,1), axis=1).tolist()

	if dataset_type == 'training':
		targets = data_df.loc[:, data_df.columns == 'target'].apply(_digit_to_10array, axis=1).tolist()

	else:
		targets = data_df['target'].tolist()

	data_list = list(zip(data, targets))

	return data_list

def _digit_to_10array(x):
	"""
	input is a digit (0-9) that gets "converted" to a numpy.ndarray of shape (10,1)
	"""
	e = np.zeros((10, 1))
	e[x] = 1.0
	return e

if __name__ == '__main__':
	datasets_dict = load_mnist('')

	# print each dataset's shape
	for dataset_type in datasets_dict:
		print(dataset_type, datasets_dict[dataset_type].shape)
	print('\n')

	training_data = datasets_dict['train']
	validation_data = datasets_dict['validation']
	test_data = datasets_dict['test']

	#_display_digits(datasets_dict)
	#_display_digits(datasets_dict, plot_type='same_digit', digit=8)
	#_display_digits_distrib(datasets_dict)

	# transform each dataframe into a numpy.ndarray to be used by the neural net
	training_data_list = _df_to_ndarray(training_data, dataset_type='training')
	test_data_list = _df_to_ndarray(test_data, dataset_type='test')
	validation_data_list = _df_to_ndarray(validation_data, dataset_type='validation')

	nn = neuralnet.NeuralNet([784, 30, 10])
	nn._stochastic_gd(training_data_list, 30, 10, 1.0, validation_data=validation_data_list)

	n_test = len(validation_data)
	test_data_eval = nn.evaluate(test_data_list)
	print('evaluation for the test dataset: {} / {}'.format(test_data_eval, n_test))

	#The values 0.1307 and 0.3081 are the global mean and standard deviation of the MNIST dataset

