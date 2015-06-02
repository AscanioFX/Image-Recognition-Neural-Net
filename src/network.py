import numpy as np
import random


class Network():

    def __init__(self, sizes):
        """
        sizes is a list containing the size of each layer. Eg. 2,3,1 means:
        1 input layer, size 2
        1 hidden layer, size 3
        1 output layers, size 1

        np.random.randn(a,b) generates a x b Numpy matrice whoose elements are
        randomly taken from the standard distribution.

        Biases and weights are stored as lists of Numpy matrices.

        self.bias is a list column numpy vector.
        self.weights list of matrices. In each matrix the i-row represent the weights of the i
        neuron.

        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Note: not assigning bias for input layer
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[1:], sizes[0:-1])]

    def feedforward(self, a):
        """ Return the output of the network if "a" is input.
            Follows the formula:

            output = sigmoid(weights.inputs + bias)

            where " . " is the matrix multiplication

            numpy.dot(a,b) is the dot product between arrays.
            For 2-D arrays it is equivalent to matrix multiplication.

            """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, test_frequency=None):
        """ Implementing stochastic gradient descent to train the network.

            training_data is a list of tuples (x, y) representing
            the training inputs and corresponding desired outputs.

            epochs is the number of epochs to train for.

            mini_batch_size is the size of the mini-batches.

            eta is the learning rate.

            optional argument test_data is for printing partial progress
            after each epoch. Since you have to evaluete the network it
            slows things down.

        """

        if test_frequency is None: test_frequency = epochs-1
        if test_data:
            n_test = len(test_data)
        else:
            n_test = 0
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            # For each epoch shuffle the training data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            # Splits training data in a list of mini-batches
            for mini_batch in mini_batches:
                self.update_mini_batch_M(mini_batch, eta)
                # Updates network's bias and weights

            log = "Epoch: {}".format(j)
            if (((j % test_frequency) == 0) and n_test > 0):
                log += ("\t\ttest: {} \ {}".format(self.evaluate(test_data), n_test))
            print log

            # Si ringrazia @lorsem per il debug di questa if


    def update_mini_batch(self, mini_batch, eta):
        """ Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.

            np.zeros returns array filled with zeros of given shape
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch_M(self, mini_batch, eta):
        """ Compute all the minibatch at the same time using a matrix of training
            input and a matrix of training output instead of looping
            over each one of them
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Matrice dei vettori di input e output affiancati (axis = 1)
        X = np.concatenate(tuple([x for x, y in mini_batch]), axis = 1)
        Y = np.concatenate(tuple([y for x, y in mini_batch]), axis = 1)

        activation = X
        activations = [activation]
        Zs = []

        for b,w in zip(self.biases, self.weights):
            # Each column of Z is the activation for a different trainig in the batch
            Z = np.dot(w, activation) + b
            Zs.append(Z)

            activation = sigmoid_vec(Z)
            activations.append(activation)

        
        delta = self.cost_derivative(activations[-1], Y) * \
            sigmoid_prime_vec(Zs[-1])

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        # now sum all columns of delta 
        delta_sum = np.expand_dims(delta.sum(axis=1), axis=1)

        nabla_b[-1] = delta_sum

        for l in xrange(2, self.num_layers):
            Z = Zs[-l]
            spv = sigmoid_prime_vec(Z)
            #print "weights[-l+1] shape: {}".format(self.weights[-l+1].transpose().shape)
            #print "delta shape: {}".format(delta.shape)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            delta_sum = np.expand_dims(delta.sum(axis=1), axis=1)
            nabla_b[-l] = delta_sum
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
       # If it did't screw up till this line, well that's a miracle.

        #learn!
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]




    def backprop(self, x, y):
        """ Return a tuple (nabla_b, nabla_w) representing the
            gradient for the cost function respect to bias and weights, calculated
            with their values.

            They are lists of numpy arrays as self.bias and self.weights
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FEEDFORWARD
        activation = x
        activations = [x]   # List to store all the activations, layer by layer
                            # first elem is the training input.
        zs = []             # List to store all the z vectors, layer by layer.

        for b, w in zip(self.biases, self.weights):
            # z is a column vector
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)

        # BACKWARD
        # Delta is the "error" of the last layer.
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])

        nabla_b[-1] = delta

        # Questo passaggio essere il male. Usa un barbatrucco numpyoso per fare questo:
        # ogni elemento della matrice nabla_w essere il prodotto tra il suo ingresso e il contributo all'errore (delta)
        # NON essere un prodotto tra matrici algebricamente legale.
        # ESEMPIO:
        # 2 ingressi : [[a1 = 1],
        #               [a2 = 2]]
        #
        # 3 uscite con errore delta : [[d1 = 2],
        #                              [d2 = 3], 
        #                              [d3 = 4]]
        #
        # Matrice nabla_w :  [[2, 4],
        #                     [3, 6],
        #                     [4, 8]]  
        #
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            # z is the activation coming from previous layer
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """ Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
            """ Relturn the vector of partial derivatives \partial C_x /
                \partial a for the output activations.
            """
            return (output_activations-y)

def sigmoid(z):
    """ Implementig the sigmoid function. """
    return 1.0/(1.0+np.exp(-z))

# Using Numpy to define a vectorized form of the sigmoid function.
# Takes numpy arrays as inputs and returns a numpy array as output.
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """ Derivative of the sigmoid function. """
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)
