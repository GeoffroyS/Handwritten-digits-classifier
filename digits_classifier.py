import os
import numpy as np
import struct

def load_mnist (path, kind='train'):
	# load from ubyte files
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

	return pixels, labels

pixels, labels = load_mnist('')
print(labels.shape) # (60000,)
print(pixels.shape) # (60000, 784): the images are squares of 28*28px = 784px
