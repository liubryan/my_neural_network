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
            out_as = self.forward_pass(data, True)
            choice = np.argmax(out_as)
            expected = np.argmax(data[1])
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
            #nabla_w and nabla_b must be reset to 0's at the end of each batch
            for batch in batches:
                nabla_w = [np.zeros(np.shape(x)) for x in self.weights]
                nabla_b = [np.zeros(np.shape(y)) for y in self.biases]               
                self.gradient_descent(batch, eta, nabla_w, nabla_b)
            if show_progress:
                print "Epoch {} Training Complete: ".format(i)
                self.working(test_set)
            if show_end_accuracy and i == (epochs-1):
                print "Finished Network: "
                self.working(test_set)


    #One image
    def gradient_descent(self, batch, eta, nabla_w, nabla_b):
        #Go through all images in the batch, summing nabla_w and nabla_b 
        #each iteration
        for single in batch:
            a_s = self.forward_pass(single)
            delta = self.cost_prime(a_s[-1], single[1]) * self.activation_prime(a_s[-1])
            nabla_w, nabla_b = self.backprop(delta, a_s, nabla_w, nabla_b)
        for i in xrange(self.num_layers-1):
            self.weights[i] -= (eta/len(batch))*nabla_w[i]
            self.biases[i] -= (eta/len(batch))*nabla_b[i]



    def forward_pass(self, datapair, get_out_as=False):
        activations = []
        activations.append(datapair[0])
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            activations.append(self.squishify(z))
        if get_out_as:
            return activations[-1]
        return activations


    def backprop(self, delta, a_s, nabla_w, nabla_b):
        nabla_w[-1] += np.dot(delta, np.transpose(a_s[-2]))
        nabla_b[-1] += delta
        for i in xrange(self.num_layers-2, 0, -1):
            delta = np.dot(np.transpose(self.weights[i]), delta) * self.activation_prime(a_s[i])
            nabla_w[i-1] += np.dot(delta, np.transpose(a_s[i-1]))
            nabla_b[i-1] += delta
        return nabla_w, nabla_b



    @staticmethod
    def squishify(zs):
        return 1.0 / (1.0 + np.exp(-zs))


    @staticmethod
    def activation_prime(activations):
        return activations * (1.0 - activations)


    #Cross-Entropy Cost Function for one run expected value 
    #from data MUST BE ARRAY
    @staticmethod
    def cost(out_a, expected):
        cost_array = 0.5*(out_a-expected)*(out_a-expected)
        return np.sum(cost_array)


    @staticmethod
    def cost_prime(out_a, expected):
        return out_a - expected