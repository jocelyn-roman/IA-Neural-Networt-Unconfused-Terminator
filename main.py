"""
Created on Apr 27, 2018

@author: J&J
"""

from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
# import os # in case of file saving


class NeuralNetwork(object):
    # This class was initially based on:
    # https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

    def __init__(self, inputs, hidden1, hidden2, output):
        # Parameters and initializations
        self.model = {}
        self.model['W1'] = np.random.randn(inputs, hidden1)/np.sqrt(inputs)
        self.model['W2'] = np.random.randn(hidden1, hidden2)/np.sqrt(hidden1)
        self.model['W3'] = np.random.randn(hidden2, output)/np.sqrt(hidden2)

    def forward(self, x):
        # Forward propagation through our network
        # dot product from the last result and W_i passed through the activation function
        out_product1 = np.dot(x, self.model['W1'])
        self.out_activation1 = self.relu(out_product1)

        out_product2 = np.dot(self.out_activation1, self.model['W2'])
        self.out_activation2 = self.relu(out_product2)

        out_product3 = np.dot(self.out_activation2, self.model['W3'])
        out_activation3 = self.stable_softmax(out_product3)


        return out_activation3

    def backward(self, x, y, output, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = y * self.one_hot_cross_entropy_prime_with_softmax(output)

        hidden2_error = output_delta.dot(self.model['W3'].T)
        hidden2_delta = hidden2_error * self.relu_prime(self.out_activation2)

        hidden1_error = hidden2_delta.dot(self.model['W2'].T)
        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)

        self.model['W3'] = self.out_activation2.T.dot(output_delta) * learning_rate
        self.model['W2'] = self.out_activation1.T.dot(hidden2_delta) * learning_rate
        self.model['W1'] = x.T.dot(hidden1_delta) * learning_rate


    # ReLU functions from https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

    @staticmethod
    def relu(x):
        # Rectified Linear Units (ReLU) activation function
        # return np.maximum(x, 0, x) # it modifies x, which is the reference
        return x * (x > 0)

    @staticmethod
    def relu_prime(x):
        # Derivative of Rectified Linear Units (ReLU)
        return 1 * (x > 0)

    @staticmethod
    def to_one_hot(y):
        # Solution based on: https://stackoverflow.com/questions/29831489/numpy-1-hot-array
        one_hot_vector = np.zeros((y.size, y.max() + 1))
        one_hot_vector[np.arange(y.size), y] = 1
        return one_hot_vector

    @staticmethod
    def softmax(x):
        # Based on: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def stable_softmax(x):
        # Based on: https://deepnotes.io/softmax-crossentropy
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp, axis=1, keepdims=True)

    # Cross-Entropy solution fetched from: Solution based on: https://deepnotes.io/softmax-crossentropy

    @staticmethod
    def cross_entropy(x, y):
        # X is the output from fully connected layer (num_examples x num_classes)
        # y is labels (num_examples x 1)

        m = y.shape[0]
        p = NeuralNetwork.softmax(x)
        log_likelihood = -np.log(p[range(m), y])
        print(p[range(m), y])
        print(log_likelihood)
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def delta_cross_entropy(x, y):
        # X is the output from fully connected layer (examples x classes)
        # y is labels (examples x 1)

        m = y.shape[0]
        grad = NeuralNetwork.softmax(x)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

    # Custom one-hot vector cross-entropy functions

    @staticmethod
    def one_hot_cross_entropy(p, q):
        # p are the one-hot labels
        # q is the result vector from softmax
        return -np.sum(p * np.log(q), axis=1)

    @staticmethod
    def one_hot_cross_entropy_prime_with_softmax(p, q):
        # p are the one-hot labels
        # q is the result vector from softmax
        return q - p

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)


def visualize_image(W, loss, title, i):
    # Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib
    element = W[:, i]
    img = element.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title("W " + str(i) + "th with loss of " + str(loss))

    # Uncomment this to show the image
    plt.show()

    # Uncomment the following code to save into disk
    '''
    directory = os.path.abspath("output/" + title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/img" + str(i))
    '''


def main():
    # Loading MNIST data set
    data = MNIST("./MNIST_data_set")
    images, labels = data.load_training()

    # Converting to numpy arrays
    labels = np.array(labels)
    images = np.array(images)

    # Getting dimensions
    first_layer = images.shape[1]
    last_layer = labels.max() + 1

    a = np.array([
        [-0.13916012, -0.15914156, -0.03611553, -0.06629650],
        [-0.25373585, 0.39812677, -0.24083797, -0.17328009],
        [-0.12787567, 0.14076882, -0.36499643, -0.32951989],
        [0.24145116, -0.01344613, 0.25512426, -0.31819186],
        [-0.02645782, 0.56205276, 0.05822283, -0.19174236],
        [0.11615288, -0.20608460, 0.05785365, -0.24800982]])

    neural_network = NeuralNetwork([first_layer, 50, last_layer])
    neural_network.forward(a)


def test():
    a = np.array([
       [-0.13916012, -0.15914156, -0.03611553, -0.06629650],
       [-0.25373585,  0.39812677, -0.24083797, -0.17328009],
       [-0.12787567,  0.14076882, -0.36499643, -0.32951989],
       [ 0.24145116, -0.01344613,  0.25512426, -0.31819186],
       [-0.02645782,  0.56205276,  0.05822283, -0.19174236],
       [ 0.11615288, -0.20608460,  0.05785365, -0.24800982]])

    b = np.array([0, 1, 2, 1, 2, 0])

    c = np.array([3, 1, 2, 1, 2, 0])

    #print(NeuralNetwork.forward(a))

    print(a.shape)
    print(NeuralNetwork.relu(a))
    print(NeuralNetwork.relu_prime(a))

    print(b.shape)
    print(NeuralNetwork.to_one_hot(b))

    print(NeuralNetwork.softmax(a))
    print(NeuralNetwork.stable_softmax(a))
    print(np.sum(NeuralNetwork.stable_softmax(a), axis=1))

    print(NeuralNetwork.cross_entropy(a, c))

    print(NeuralNetwork.one_hot_cross_entropy(NeuralNetwork.to_one_hot(c), NeuralNetwork.softmax(a)))
    print(np.mean(NeuralNetwork.one_hot_cross_entropy(NeuralNetwork.to_one_hot(c), NeuralNetwork.softmax(a))))

    print(NeuralNetwork.delta_cross_entropy(a, c))  # Different because m division
    print(NeuralNetwork.one_hot_cross_entropy_prime_with_softmax(NeuralNetwork.to_one_hot(c), NeuralNetwork.softmax(a)))


if __name__ == "__main__":
     main()
    #test()
