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
			X_train = X_train.applymap(lambda x: x/255)
			X_validation = X_validation.applymap(lambda x: x/255)
			#print(X_train.loc[0].tolist())
			df_train = pd.concat([X_train, y_train], axis=1, sort=False)
			df_validation = pd.concat([X_validation, y_validation], axis=1, sort=False)
			df_dict["train"] = df_train
			df_dict["validation"] = df_validation
		else:
			df.loc[:, df.columns != 'target'].applymap(lambda x: x/255)
			df_dict["test"] = df 

	return df_dict

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
	df_train = datasets_dict["train"]
	digits_dict = {}
	for i in range(10):
		digits_dict[i] = df_train.loc[df_train['target'] == i].shape[0]
		print('number of instances of {} in dataset: {}'
			.format(i, digits_dict[i]))

	x = [key for key in digits_dict]
	y = [val for val in digits_dict.values()]

	plt.bar(x, y)
	plt.show()

def _df_to_ndarray(data_df, dataset_type):
	data = data_df.loc[:, data_df.columns != 'target'].apply(lambda row: row.to_numpy(dtype='float32').reshape(784,1), axis=1).tolist()

	if dataset_type == 'training':
		targets = data_df.loc[:, data_df.columns == 'target'].apply(_digit_to_10array, axis=1).tolist()

	else:
		targets = data_df['target'].tolist()

	data_list = list(zip(data, targets))

	return data_list

def _digit_to_10array(x):
	e = np.zeros((10, 1))
	e[x] = 1.0
	return e

if __name__ == '__main__':
	datasets_dict = load_mnist('')
	for dataset_type in datasets_dict:
		print(dataset_type, datasets_dict[dataset_type].shape)
	print('\n\n\n')
	training_data = datasets_dict['train']
	validation_data = datasets_dict['validation']
	test_data = datasets_dict['test']
	#_display_digits(datasets_dict)
	#_display_digits(datasets_dict, plot_type='same_digit', digit=8)
	#_display_digits_distrib(datasets_dict)

	training_data_list = _df_to_ndarray(training_data, dataset_type='training')
	test_data_list = _df_to_ndarray(test_data, dataset_type='test')
	validation_data_list = _df_to_ndarray(validation_data, dataset_type='validation')
	#print(validation_data_list[0])
	
	# print(" Using the 'test' data\n",
	# 	"this should be a list/size 10000: ", type(data_list), len(data_list), "\n",
	# 	"this should be tuple/size 2: ", type(data_list[0]), len(data_list[0]), "\n", 
	# 	"this should be ndarray/size (784,): ", type(data_list[0][0]), data_list[0][0].shape, data_list[0][0], "\n",
	# 	"this should be ndarray/size (10,): ", type(data_list[0][1]), data_list[0][1].shape, data_list[0][1], "\n"
	# 	)


	nn = neuralnet.NeuralNet([784, 30, 10])
	nn._stochastic_gd(training_data_list, 30, 10, 3.0, validation_data=validation_data_list)



