"""basic neural network"""
"""USAGE: this neural network is designed to take in the MNIST gzip pickled
dataset of handwritten numbers from 0 to 9. To run, download this file and the
MNIST dataset at http://deeplearning.net/tutorial/gettingstarted.html. Edit the
gzip.open line below to where the dataset is located for you. At the command
line in Python, type 
    "import neural_network as n"
    "train, valid, test = n.init_data()"
    "net = n.Network([784, 30, 10])"
    "net.training(train, test, 10, 3.0, 20, True)"
At this point, the network should be training. You can play around with the 
number of hidden neurons, hidden layers, eta, batch size, and number of training
epochs if you want.
"""



import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cPickle, gzip



"""Initialize data from MNIST dataset. Separates training,
validating, and testing set. Data from each is processed into a list of pairs 
(x,y) where x is the pixel input of the number, y is an array corresponding to 
the expected output activations of the network"""
def init_data():
    #Specify where the dataset is located on your computer
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    raw_train, raw_valid, raw_test = cPickle.load(f)
    f.close()

    train_set = format_data(raw_train)
    valid_set = format_data(raw_valid)
    test_set = format_data(raw_test)

    print "train_set has {} images\n valid_set has {} images\n test_set has {} images\n".format(
            len(train_set), len(valid_set), len(test_set))
    return train_set, valid_set, test_set


"""Converts MNIST dataset from a pair of list of images and output labels
into a list of pairs of images and their respective labels converted
into their equivalent representation as output layer neurons"""
def format_data(data):
    pixel_in = [np.reshape(x, (784,1)) for x in data[0]]
    expected = []
    for i in data[1]:
        arr = np.zeros((10,1))
        arr[i] = 1.0
        expected.append(arr)
    return zip(pixel_in, expected)


"""
np.random.seed(5678923)
fig, ax = ax.plot(np.random.rand(190))
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0,1)

def update(data):
    line.set_ydata(data)
    return line,

def data_gen():
    while True:
        yield np.random.rand(10)
"""


"""REQUIRED: Number of input layer neurons must be the same as the number
of pixels of the dataset image. Number of output neurons correspond to 
the total number of different outputs"""
class Network(object):


    #Weights and biases are a list of arrays, where each array represents
    #its corresponding neuron layer from L = 2 to N (layer 1 does not
    #have weights or biases)
    def __init__(self, size):
        self.size = size
        self.num_layers = len(size)
        self.biases = [np.random.randn(y,1) for y in size[1:]]
        self.weights = [np.random.randn(y,x) / np.sqrt(x)
                        for x,y in zip(size[:-1], size[1:])]



    #Tests neural net's accuracy on new data after each epoch of training
    def working(self, test_set, func_type):
        num_correct = 0
        for data in test_set:
            if (func_type == "squared"):
                out_as = self.forward_pass(data, "sigmoid", True)
            else:
                out_as = self.forward_pass(data, "softmax", True)
            choice = np.argmax(out_as)
            expected = np.argmax(data[1])
            if choice == expected:
                num_correct += 1
        print "{}/{} correctly classified by network\n".format(
            num_correct, len(test_set))



 

    #Trains the neural net using backprop. At the end of each epoch 
    #the test data is used to evaluate the network's accuracy
    def training(self, train_set, test_set, batch_size, eta, epochs, 
                cost_activation_type, show_progress=True):
        if (cost_activation_type != "squared" and cost_activation_type != "cross"):
            print("enter valid cost_activation_type 'squared' or 'cross'")
            exit(1)
        for i in xrange(epochs):     
            random.shuffle(train_set)
            #Separates the dataset into batch sizes specified by the user
            batches = [train_set[k:batch_size+k] 
                    for k in xrange(0,len(train_set),batch_size)]
            for batch in batches:
                #nabla_w and nabla_b are reset before each batch 
                nabla_w = [np.zeros(np.shape(x)) for x in self.weights]
                nabla_b = [np.zeros(np.shape(y)) for y in self.biases]               
                self.gradient_descent(batch, eta, nabla_w, nabla_b, cost_activation_type)
            if show_progress:
                print "Epoch {} Training Complete: ".format(i)
                self.working(test_set, cost_activation_type)


    #Determines the weight and bias errors after each backprop iteration of 
    #a batch. At the end of a batch, the errors are averaged and used to 
    #calculate the new weights and biases. Uses cross-entropy cost function
    def gradient_descent(self, batch, eta, nabla_w, nabla_b, func_types):
        if func_types == "squared":
            for single in batch:
                a_s = self.forward_pass(single, "sigmoid")
                delta = self.cost_prime(a_s[-1], single[1], "squared") * \
                    self.activation_prime(a_s[-1], "sigmoid")
                nabla_w, nabla_b = self.backprop(delta, a_s, nabla_w, nabla_b)
        else:
            for single in batch:
                a_s = self.forward_pass(single, "softmax")
                #delta = self.cost_prime(a_s[-1], single[1], "cross") * \
                    #self.activation_prime(a_s[-1], "softmax")
                delta = a_s[-1] - single[1]
                nabla_w, nabla_b = self.backprop(delta, a_s, nabla_w, nabla_b)            
        for i in xrange(self.num_layers-1):
            self.weights[i] -= (eta/len(batch))*nabla_w[i]
            self.biases[i] -= (eta/len(batch))*nabla_b[i]


#TODO MAKE ONLY OUTPUT ACTIVATION BE SOFTMAX
    #Calculates and returns the activations of all the neurons
    def forward_pass(self, datapair, func_type, get_out_as=False):
        activations = []
        activations.append(datapair[0])
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            activations.append(self.squishify(z, func_type))
        if get_out_as:
            return activations[-1]
        return activations



    #Backpropagates the error from the last layer to calculate and update nabla_w and
    #nabla_b for layers 2 to N
    def backprop(self, delta, a_s, nabla_w, nabla_b):
        nabla_w[-1] += np.dot(delta, np.transpose(a_s[-2]))
        nabla_b[-1] += delta
        for i in xrange(self.num_layers-2, 0, -1):
            delta = np.dot(np.transpose(self.weights[i]), delta) * self.activation_prime(a_s[i], "sigmoid")
            nabla_w[i-1] += np.dot(delta, np.transpose(a_s[i-1]))
            nabla_b[i-1] += delta
        return nabla_w, nabla_b



    #Activation function, takes in zs of a layer
    @staticmethod
    def squishify(zs, choice):
        if (choice == "sigmoid"):
            return 1.0 / (1.0 + np.exp(-zs))
        elif (choice == "softmax"):
            #normalizes softmax so overflow does not occur
            normalize = np.max(zs)
            exps = np.exp(zs - normalize)
            return exps / np.sum(exps)
        else:
            print("incorrect squishify choice. Choices are 'sigmoid' and 'softmax'\n")
            exit(1)

    #Activation derivative (sigmoid)
    @staticmethod
    def activation_prime(activations, choice):
        if (choice == "sigmoid"):
            return activations * (1.0 - activations)
        elif (choice == "softmax"):
            return activations
            """
            a_prime = np.diag(activations)
            for i in xrange(np.size(activations)):
                for j in xrange(np.size(activations)):
                    if i == j:
                        a_prime[i][j] = activations[i] * (1 - activations[i])
                    else:
                        a_prime[i][j] = -activations[i] * activations[j]
            return a_prime
            """
        else:
            print("incorrect a_prime choice. Choices are 'sigmoid' and softmax'\n")
            exit(1)





    #COST FUNCTIONS AND THEIR DERIVATIVES
    #----------------------------------------------------------------
    #DIFFERENCE SQUARED
    @staticmethod
    def squared_cost(out_a, expected):
        cost_array = 0.5*(out_a-expected)*(out_a-expected)
        return np.sum(cost_array)


    #CROSS ENTROPY
    @staticmethod
    def cross_entropy_cost(out_a, expected):
        cost_array = expected*np.log(out_a) + (1.0-expected)*np.log(1.0-out_a)
        return np.sum(cost_array)
     

    #dC/da cost derivative
    @staticmethod
    def cost_prime(out_a, expected, choice):
        if (choice == "cross"):
            return (out_a - expected) / (out_a * (1.0-out_a))
            #when using softmax 
        elif (choice == "squared"):
            #When using sigmoid activation function
            return out_a - expected
        else:
            print("incorrect cost_prime choice. Choices are 'cross' and 'squared\n")
            exit(1)

