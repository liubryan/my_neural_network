"""basic neural network"""


"""activations list - Layers 1,2,...,L (size = L) 
biases, zs, activations, deltas, *nablas - Layers 2,3,...,L (size = L-1)"""

import numpy as np
import random
import math
import cPickle, gzip


"""Initialize data from MNIST gzip pickled number dataset. Converts split
data into a list of pairs (x,y) where x is the pixel input of the number, 
y is the integer label for that image"""
def init_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    raw_train, raw_valid, raw_test = cPickle.load(f)
    f.close()

    train_set = format_data(raw_train)
    valid_set = format_data(raw_valid)
    test_set = format_data(raw_test)

    print "train_set has {} images\n valid_set has {} images\n test_set has {} images\n".format(
            len(train_set), len(valid_set), len(test_set))
    return train_set, valid_set, test_set



def format_data(data):
    pixel_in = [np.reshape(x, (784,1)) for x in data[0]]
    expected = []
    for i in data[1]:
        arr = np.zeros((10,1))
        arr[i] = 1.0
        expected.append(arr)
    return zip(pixel_in, expected)





"""REQUIRED: Number of input layer neurons must be the same as the number
of pixels of the dataset image. Number of output neurons must be 10 to
represent all numbers 0 to 9"""
class Network(object):

    def __init__(self, size):
        self.size = size
        self.num_layers = len(size)
        self.biases = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(y,x) #/ np.sqrt(x)
                        for x,y in zip(size[:-1], size[1:])]


    def working(self, test_set):
        num_correct = 0
        for data in test_set:
            _,expected = data
            out_as = self.forward_pass(data, True)
            choice = np.argmax(out_as)
            if choice == expected:
                num_correct += 1
        print "{}/{} correctly classified by network\n".format(
            num_correct, len(test_set))


    #Automates choosing eta and batch size
    #def picking_hyper(self, valid_set): 

    def training(self, train_set, test_set, batch_size, eta, epochs, 
                show_progress=False, show_end_accuracy=False):
        for i in xrange(epochs):        
            random.shuffle(train_set)
            #list of lists of training data
            batches = [train_set[k:batch_size+k] 
                    for k in xrange(0,len(train_set),batch_size)]
            #Since deltas are overwritten at each new image, I just initialized
            #the deltas for the entire train_set to avoid uneccesary re-inits. 
            #nabla_w and nabla_b must be reset to 0's at the end of each batch
            #deltas = [np.zeros(np.shape(a)) for a in self.biases]
            for batch in batches:
                deltas = [np.zeros(np.shape(a)) for a in self.biases]
                nabla_w = [np.zeros(np.shape(a)) for a in self.weights]
                nabla_b = [np.zeros(np.shape(a)) for a in self.biases]
                self.gradient_descent(batch, deltas, nabla_w, nabla_b, eta)
            if show_progress:
                print "Epoch {} Training Complete: ".format(i)
                self.working(test_set)
            if show_end_accuracy and i == (epochs-1):
                print "Finished Network: "
                self.working(test_set)


    #One image
    def gradient_descent(self, batch, delta, nabla_w, nabla_b, eta):
        #Go through all images in the batch, updating nabla_w and nabla_b 
        #each iteration by adding
        for single in batch:
            activations, activations_prime, cost, cost_prime = self.forward_pass(single)
            self.backprop(delta, activations, activations_prime, cost_prime, 
                    nabla_w, nabla_b, single[0])
        for i in xrange(self.num_layers-1):
            self.weights[i] -= (eta/len(batch))*nabla_w[i]
            self.biases[i] -= (eta/len(batch))*nabla_b[i]


    def forward_pass(self, datapair, get_out_as=False):
        #Initialize zs and activations list of arrays from layer = 2 to end
        zs = [np.zeros(np.shape(a)) for a in self.biases]
        activations = zs
        activations_prime = zs
        for i in xrange(self.num_layers - 1):
            if i != 0:
                zs[i] = np.dot(self.weights[i], activations[i-1]) + self.biases[i]
            else:
                zs[i] = np.dot(self.weights[i], data_in) + self.biases[i]
            activations[i], activations_prime[i] = self.squishify(zs[i])
        if get_out_as:
            return activations[-1]
        cost, cost_prime = self.cost_func(activations[-1], datapair[1])
        return activations, activations_prime, cost, cost_prime


    def backprop(self, deltas, activations, activations_prime, cost_prime, 
        nabla_w, nabla_b, data_in):
        deltas[-1] = cost_prime * activations_prime[-1]
        nabla_w[-1] += np.dot(deltas[-1], np.transpose(activations[-2]))
        nabla_b[-1] += deltas[-1]
        #Iterate from layers L-1 to 2. For some layer, 
        #activations[i+1], weights[i], deltas[i]                
        for i in reversed(xrange(len(deltas)-1)):
            deltas[i] = np.dot(np.transpose(self.weights[i+1]), deltas[i+1]) * \
                activations_prime[i]
            if i != 0:
                nabla_w[i] += np.dot(deltas[i], np.transpose(activations[i-1]))
            else:
                nabla_w[i] += np.dot(deltas[i], np.transpose(data_in))
            nabla_b[i] += deltas[i] 


    @staticmethod
    def squishify(zs):
        sigmoid = 1.0 / (1.0 + np.exp(-zs))
        sigmoid_prime = sigmoid * (1.0 - sigmoid)
        return sigmoid, sigmoid_prime


    #Cross-Entropy Cost Function for one run expected value 
    #from data MUST BE ARRAY
    @staticmethod
    def cost_func(out_a, expected):
        cost_array = 0.5*(out_a-expected)*(out_a-expected)
        cost = np.sum(cost_array)
        cost_prime = out_a - expected
        return cost, cost_prime