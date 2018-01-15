"""basic neural network"""


"""activations list - Layers 1,2,...,L (size = L) 
biases, zs, activations, deltas, *nablas - Layers 2,3,...,L (size = L-1)"""

import numpy as np
import random
import math
import cPickle, gzip

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print "train_set has {} images\n valid_set has {} images\n test_set has {} images\n\n".format(
		len(train_set), len(valid_set), len(test_set))






class Network(object):

	def __init__(self, size):
		self.size = size
		self.num_layers = len(size)
		self.biases = [np.random.randn(y,1) for y in size[1:]]
		self.weights = [np.random.randn(y,x) / sqrt(x)
						for x,y in zip(size[:-1], size[1:])]


	def working(self, test_set):
		num_correct = 0
		for data in test_set:
			out_as = forward_pass(data, 0, get_out_as=True)
			choice = np.argmax(out_as)
			if(choice == data[1]):
				num_correct += 1
		print "{}/{} correctly classified by network\n".format(
				num_correct, len(test_set))


"""
	#Automates choosing eta and batch size
	def picking_hyper(self, valid_set):
"""


	def training(self, train_set, test_set, batch_size, eta, epochs, 
				show_progress=False, show_end_accuracy=False):
		for i in xrange(epochs):		
			random.shuffle(train_set)
			#list of lists of training data
			batches = [train_data[k::batch_size] for k in xrange(batch_size)]
			"""Since deltas are overwritten at each new image, I just initialized
			the deltas for the entire train_set to avoid uneccesary re-inits. 
			nabla_w and nabla_b must be reset to 0's at the end of each batch."""
			deltas = [np.zeros(np.shape(a)) for a in self.biases]
			for batch in batches:
				nabla_w = [np.zeros(np.shape(a)) for a in self.biases]
				nabla_b = [np.zeros(np.shape(a)) for a in self.biases]
				gradient_descent(batch, deltas, nabla_w, nabla_b, eta)
			if show_progress:
				print "Epoch {} Training Complete: ".format(i)
				working(test_set)
			if show_end_accuracy and i == (epochs-1):
				print "Finished Network: "
				working(test_set)


	#One image
	def gradient_descent(self, batch, delta, nabla_w, nabla_b, eta):
		"""Go through all images in the batch, updating nabla_w and nabla_b 
		each iteration by adding"""
		for single in batch:
			activations, activations_prime, cost, cost_prime = forward_pass(single)
			backprop(delta, activations, activations_prime, cost_prime, 
					nabla_w, nabla_b)
		self.weights = [self.weights[i] - (eta/size(batch))*nabla_w[i]
						for i in xrange(self.num_layers-1)]
		self.biases = [self.biases[i] - (eta/size(batch))*nabla_b[i]
						for i in xrange(self.num_layers-1)]


	def forward_pass(self, datapair, get_out_as=False):
		#data_in must be an vertical array. Expected converted to array of 
		#output activations
		data_in, expected = data
		expected_as = np.shape(self.biases[-1])
		expected_as[expected] = 1.0
		#Initialize zs and activations list of arrays from layer = 2 to end
		zs = [np.zeros(np.shape(a)) for a in self.biases]
		activations = zs
		activations_prime = zs
		for i in xrange(self.num_layers - 1):
			if i != 0:
				zs[i] = np.dot(self.weights[i], activations[i]) + self.biases[i]
			else:
				zs[i] = np.dot(self.weights[i], data_in) + self.biases[i]
			activations[i], activations_prime[i] = squishify(zs[i])
		cost, cost_prime = cross_entropy_cost(activation[-1], expected_as)
		if np.argmax(activations[-1]) == expected:
			num_correct += 1
		if get_out_as:
			return activations[-1]
		else:
			return activations, activations_prime, cost, cost_prime


	def backprop(self, deltas, activations, activations_prime, cost_prime, 
				nabla_w, nabla_b):
		deltas[-1] = cost_prime * activations_prime[-1]
		nabla_w += np.dot(delta[-1], activations[-2].transpose())
		nabla_b[-1] += delta[-1]
		#Iterate from layers L-1 to 2. For some layer, 
		#activations[i+1], weights[i], deltas[i]				
		for i in reversed(xrange(len(delta)-1)):
			delta[i] = np.dot(self.weights[i+1].transpose(), deltas[i+1])
						* activations_prime[i+1]
			nabla_w[i] += np.dot(deltas[i], activations[i].transpose())
			nabla_b[i] += deltas[i]	


	"""softmax for all zs in a layer for one run"""
	"""
	@staticmethod
	def squishify(zs):
		e_x = np.exp(zs)
		sum_e_x = np.sum(e_x)
		softmax = e_x / sum_e_x
		softmax_prime = []
		for i in zs:
			for j in zs:
				if (i == j):
	"""

	@staticmethod
	def squishify(zs):
		sigmoid = 1.0 / (1.0 + np.exp(-zs))
		sigmoid_prime = activations * (1.0 - activations)
		return sigmoid, sigmoid_prime


	#Cross-Entropy Cost Function for one run expected value 
	#from data MUST BE ARRAY
	@staticmethod
	def cross_entropy_cost(output_a, expected):
		cost_array = expected*np.log(out_a) + (1.0-expected)*np.log(1.0-out_a)
		cost = np.sum(cost_array)
		cost_prime = (out_a - expected) / (out_a * (1.0-out_a))
		return cost, cost_prime