import os
import struct

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist (path, kind='train'):
	# load ubyte files downloaded from Yann LeCun's page
	# http://yann.lecun.com/exdb/mnist/

	for kind in ['train', 'test']
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

		if kind eq 'train':
			X = df[:,:-1]
			y = df['target']
			X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=0)

	#return df

#print(load_mnist('').shape) # (60000, 785): 60k rows 
							# and 28 by 28 = 784px images + 'target' column


# if validation_set and kind == 'train':
# 	X_train, X_val, y_train, y_val = train_test_split(df.iloc[:,:-1],
# 		df['target'],
# 		train_size=50000,
# 		random_state=0)
# 	df = X_val
# 	df['target'] = y_val