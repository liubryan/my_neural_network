"""basic neural network"""

import numpy as np
import random
import math
import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


class Network(object):

	def __init__(self, size):
		self.size = size
		self.num_neurons = size
		self.biases = [np.random.randn(y,1) for y in size[1:]]
		self.weights = [np.random.randn(y,x) / sqrt(x)
						for x,y in zip(size[:-1], size[1:])


	def testing(self, train_data, batch_size, epochs):
		for i in xrange(epochs):
			training(train_data, batch_size)


	def working(self, datapair):
		data_in, expected = datapair
		activations = [np.zeros(shape(a)) for a in self.biases]
		output_activations, cost = forward_pass(data_in)
		backward_pass(output_activations, cost)


	def training(self, train_set, batch_size):
		random.shuffle(training_data)
		#list in list
		batches = [train_data[k::batch_size] for k in xrange(batch_size)]
		for batch in batches:
			for batch[i] in xrange(batch):


	"""softmax for all z"""
	@staticmethod
	def squishify(z)
		e_x = [np.exp(z) for z in a.biases]
		sum_e_x = [np.sum(layer) for layer in e_x]
		activations = []
		for i,j in zip(e_x, sum_e_x):
			activation.append(i/j)
		return activations

	"""Cross-Entropy Cost Function for the mini batch"""
	@staticmethod
	def cost(output_activations, expected, mini_batch_size):
		cost = []
		for x, y in zip(output_activations, expected):
			cost += y*np.log(x) + (1.0-y)*np.log(1.0-x)
		return (-1.0/mini_batch_size) * cost




