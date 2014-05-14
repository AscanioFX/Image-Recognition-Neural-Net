#
# Imports
#

import numpy as np

class BackPropagationNetwork:
    """ A back-prpagation network """

    #
    # Class members
    #

    layerCount = 0
    shape = None
    weights = []

    #
    # Class methods
    #

    def __init__(self, layerSize):
        """ Initialize the network """

        # Layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        # Imput/Output data from last run
        self._layerImput = []
        self._layerOutput = []

        # Create weight-arrays
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 0.1, size = (l2, l1+1)))

#
# If run as a script create a test object
#

if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
