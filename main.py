"""
Created on Apr 27, 2018

@author: J&J
"""

import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_sizes, output_size):
        # parameters
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.z = []

        # weights
        for i in range(len(self.sizes) - 1):
            self.weights.append(np.random.rand(self.sizes[i], self.sizes[i+1]))

    def forward(self, x):
        # forward propagation through our network
        self.z = [x]
        for W_i in self.weights:
            # dot product from the last result and W_i
            product = np.dot(self.z[-1], W_i)
            # result passed through the activation function
            self.z.append(self.relu(product))

        # the output is the last Z
        output = self.z[-1]
        return output

    def backward(self, x, y, o):
        return

    # ReLU functions from https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

    @staticmethod
    def relu(x):
        # Rectified Linear Units (ReLU) activation function
        return np.maximum(x, 0, x)

    @staticmethod
    def relu_prime(x):
        # derivative of Rectified Linear Units (ReLU)
        return 1. * (x > 0)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)


def main():
    x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)

    # scale units
    x = x / np.amax(x, axis=0)  # maximum of X array
    y = y / 100  # max test score is 100

    neural_network = NeuralNetwork(2, [3, 5, 10, 2], 1)

    for i in range(1):  # trains the NN 1,000 times
        print("Input: \n" + str(x))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(neural_network.forward(x)))
        print("Loss: \n" + str(np.mean(np.square(y - neural_network.forward(x)))))  # mean sum squared loss
        print("\n")
        neural_network.train(x, y)


if __name__ == "__main__":
    main()
