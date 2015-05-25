import mnist_loader
import network

#
# Importing data from mnist database
#

print "Loading data...."
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "Data loaded."

#
# Initializing network
#

# 1 hidden layer network
input_neurons = 784
hidden_neurons = 5
output_neurons = 10
test_frequency = 1

eta = 0.35
epochs = 100
mini_batch_size = 10

print "Initializing network..."
print "Hiidden neurons: {0}".format(hidden_neurons)
print "Eta: {0} \nEpochs: {1} \nBatch size: {2}\nFreq: {3}\n\n".format(eta, epochs, mini_batch_size, test_frequency)
net = network.Network([input_neurons, hidden_neurons, output_neurons])
net.SGD(training_data[:1000], epochs, mini_batch_size, eta, test_data, test_frequency)
