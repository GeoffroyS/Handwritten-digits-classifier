import os
import struct

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist (path, kind='train'):
	# load ubyte files downloaded from Yann LeCun's page
	# http://yann.lecun.com/exdb/mnist/
	# return a dictionary containing the 3 pd.DataFrames:
	# train, validation, test

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
			X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=50000, random_state=0)
			df_train = pd.concat([X_train, y_train], axis=1, sort=False)
			df_validation = pd.concat([X_validation, y_validation], axis=1, sort=False)
			df_dict["train"] = df_train
			df_dict["validation"] = df_validation
		else:
			df_dict["test"] = df

	return df_dict

datasets = load_mnist('')
for dataset_type in datasets:
	print(dataset_type, datasets[dataset_type].shape)
