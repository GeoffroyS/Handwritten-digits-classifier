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
		# np.random.seed(1)
		self.sizes = sizes
		self.number_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.best_weights = []
		self.best_biases = []

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
			a = self._sigmoid(np.dot(w, a) + b)
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
			previous_evaluation = 0

			for mini_batch in mini_batches:
				self._update_mini_batch(mini_batch, eta)
			if validation_data:
				evaluation = self.evaluate(validation_data)
				if evaluation > previous_evaluation:
					self.best_weights = self.weights
					self.best_biases = self.biases
				previous_evaluation = evaluation
				print('epoch {}: {} / {}'.format(i, evaluation, n_validation))
			else:
				print('epoch {} complete'.format(i))
		self.weights = self.best_weights
		self.biases = self.best_biases

	def _update_mini_batch(self, mini_batch, eta):
		"""
		Update the network's weights and biases by applying SGD,
		and using backpropagation
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self._backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip (self.biases, nabla_b)]
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip (self.weights, nabla_w)]

	def evaluate(self, data):
		"""
		Return the number of test inputs for which the target was guessed correctly
		The NN's output is assumed to be the index of the neuron (in the output layer)
		with the highest activation
		data is either the validation data or test data
		"""
		results = [(np.argmax(self._feedforward(x)), y) for (x, y) in data]
		return sum(int(x == y) for (x, y) in results)

	def _backprop(self, x, y):
		"""
		Return a table (nabla_b, nabla_w) representing the gradient
		for the cost function.
		Both nabla_b and nabla_w are lists of numpy arrays, one array being a layer
		of biases or weights
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x] #list storing all the activations, layer by layer
		zs = [] #list to store all the z vectors, layer by layer

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self._sigmoid(z)
			activations.append(activation)

		#backward prop
		delta = self._cost_derivative(activations[-1], y) * self._sigmoid_gradient(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# l is the last layer of neurons
		for l in range(2, self.number_layers):
			z = zs[-l]
			delta = np.dot(self.weights[-l+1].transpose(), delta) * self._sigmoid_gradient(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

	def _cost_derivative(self, output_activations, y):
		"""
		Return the vector of partial derivatives
		"""
		return (output_activations - y)
