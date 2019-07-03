"""
neuralnet.py

A module implementing the stochastic gradient descent
learning algorithm for a feedforward (Recurrent Neural Net coming up next!)
neural net.
Gradients calculated using backpropagation
"""

import random

import numpy as np

class NeuralNet(object):
	def __init__(self, sizes):
		"""
		sizes is a list containing the number or neurons in the different layers
		[10, 30, 5] would be a NN containing
			- 10 neurons in the input layer
			- 30 neurons in one hidden layer
			- 5 neurons in the output layer
		We'll start by using a random initialisation (w/ Gaussian distribution)
		of the weights and biases
		No biases for the input layer
		TODO: Compare with Xavier/He initialisations
		"""
		np.random.seed(1)
		self.sizes = sizes
		self.number_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def _sigmoid(self, z):
		return 1.0/(1.0+np.exp(-z))

	def _sigmoid_gradient(self, z):
		return self._sigmoid(z)*(1 - self._sigmoid(z))

	def _feedforward(self, a):
		"""
		a is the input
		a is an (n, 1) numpy ndarray, not a (n,) vector
		"""
		for b, w in zip(self.biases, self.weights):
			a = _sigmoid(np.dot(w, a) + b)
		return a

	def _stochastic_gd(self, training_data, epochs, mini_batch_size, eta=0.1, validation_data=None):
		"""
		Mini-batch Stochastic Gradient Descent to train the NN
		training_data is a list of tuples "(x, y)" (training input, target)
		validation_data is obviously the same format
		If validation_data: NN evaluated against it and result printed out
		This is to check that there's no overfitting
		eta is the learning rate
		"""
		if validation_data:
			n_validation = len(validation_data)
		n = len(training_data)

		for i in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
			for Mini-batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if validation_data:
				print('epoch {}: {} / {}'.format(j, self.evaluate(validation_data), n_validation))
			else:
				print('epoch {} complete'.format(j))

	def _update_mini_batch(self, mini_batch, eta):
		"""
		Update the network's weights and biases by applying SGD,
		and using backpropagation
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip (self.biases, nabla_b)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip (self.weights, nabla_w)]

	def _evaluate(self, validation_data):

	def _backprop()

