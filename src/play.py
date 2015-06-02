import mnist_loader
import network
import numpy as np

#
# Settings
#


# 1 hidden layer network
input_neurons = 784
#input_neurons = 2
hidden_neurons = 50
output_neurons = 10
test_frequency = 10
training_size = 50000
#training_size = 3


eta = 0.35
epochs = 100
mini_batch_size = 100

print "Hidden neurons: {0}".format(hidden_neurons)
print "Eta: {0} \nEpochs: {1} \nBatch size: {2}\nFreq: {3}\n\
Training set size: {4}\n\n".format(eta, epochs, mini_batch_size, test_frequency, training_size)

#
# Importing data from mnist database
#

print "Loading data...."
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "Data loaded."
#
# Initializing debug data
#

#raining_data=[(np.array([[1,1,1]]).transpose(), np.array([[1,0]]).transpose()) for i in range(3)]
#test_data = training_data
#print training_data

#
# Initializing network
#

print "Initializing network..."
net = network.Network([input_neurons, hidden_neurons, output_neurons])
print "Starting to learn..."
net.SGD(training_data[:training_size], epochs, mini_batch_size, eta, test_data, test_frequency)
