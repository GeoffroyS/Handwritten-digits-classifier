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