import matplotlib.pyplot as plt
import numpy as np
import pickle

import os


class NeuralNetwork(object):
    # This class was initially based on:
    # https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

    def __init__(self):
        # Class initializations
        self.model = dict()
        self.graph = dict()
        self.graph['loss'] = []
        self.graph['accuracy'] = []
        self.graph['epoch'] = []

    def forward(self, x):
        return np.array([])

    def forward_propagation_with_dropout(self, x, keep_prob=0.5):
        return np.array([]), int

    def backward(self, x, y, output, learning_rate=0.0085):
        pass

    def backward_propagation_with_dropout(self, x, y, output, d1, keep_prob, learning_rate=0.0085):
        pass

    def feed_backward(self, y):
        return np.array([])

    # ReLU functions from https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

    @staticmethod
    def relu(x):
        # Rectified Linear Units (ReLU) activation function
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
        # Returns a vector with loss per data
        # p are the one-hot labels
        # q is the result vector from softmax
        return -np.sum(p * np.log(q), axis=1)

    @staticmethod
    def one_hot_cross_entropy_prime_with_softmax(p, q):
        # p are the one-hot labels
        # q is the result vector from softmax
        return q - p

    @staticmethod
    def cross_entropy_loss(p, q):
        # Returns a averaged value for all data
        # p are the one-hot labels
        # q is the result vector from softmax
        return np.mean(NeuralNetwork.one_hot_cross_entropy(p, q))

    @staticmethod
    def accuracy(output, labels):
        # Gets the total element count
        total = output.shape[0]
        # Gets the difference between the indices of the maximum values
        difference = np.argmax(output, axis=1) - np.argmax(labels, axis=1)
        # Counts the correct predictions (zeros)
        correct = np.count_nonzero(difference == 0)
        return correct / total

    # Save/Load weights from: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict

    def save(self, filename):
        # Saves weights in file
        with open(filename, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        # Loads weights from file
        with open(filename, 'rb') as handle:
            self.model = pickle.load(handle)

    def plot(self, path):
        # The plot shows the learning behavior

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.graph['epoch'], self.graph['loss'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(self.graph['epoch'], self.graph['accuracy'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        # plt.show()

        directory = os.path.abspath("output/plot")
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + "/" + path)

    def train(self, x, y, batch_size, epoch):

        # Converting labels to one-hot encoding
        labels = self.to_one_hot(y)

        eighty = int(round(x.shape[0] * 0.8))

        # For each epoch
        for i in range(epoch):
            print("Epoch #", i)

            # Randomly select training and validation set by 80% - 20%
            # This is called Holdout method
            # https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Holdout_method
            # https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
            indices = np.random.permutation(x.shape[0])
            training_idx, validation_idx = indices[:eighty], indices[eighty:]
            training, validation = x[training_idx, :], x[validation_idx, :]
            training_labels, validation_labels = labels[training_idx, :], labels[validation_idx, :]

            # Uncomment following lines for non cross-validation
            # training, validation = x, x
            # training_labels, validation_labels = labels, labels

            # Calculating batch size
            total = training.shape[0]
            batches = total / batch_size

            # Dividing by mini-batches
            batch_data = np.split(training, batches, axis=0)
            batch_labels = np.split(training_labels, batches, axis=0)

            # Take each mini-batch and train
            for idx, (mini_data, mini_labels) in enumerate(zip(batch_data, batch_labels)):
                output, d1 = self.forward_propagation_with_dropout(mini_data)
                loss = self.cross_entropy_loss(mini_labels, output)
                accuracy = self.accuracy(output, mini_labels)
                # print("Loss: ", loss)
                # print("Accuracy: ", accuracy)
                self.graph['loss'].append(loss)
                self.graph['accuracy'].append(accuracy)
                self.graph['epoch'].append(i + (idx / batches))
                self.backward_propagation_with_dropout(mini_data, mini_labels, output, d1, 0.5)

            # Validating
            output = self.forward(validation)
            loss = self.cross_entropy_loss(validation_labels, output)
            accuracy = self.accuracy(output, validation_labels)
            self.graph['loss'].append(loss)
            self.graph['accuracy'].append(accuracy)
            self.graph['epoch'].append(i + 1)

    def test(self, x, y):
        # Converting labels to one-hot encoding
        labels = self.to_one_hot(y)

        # Doing feed forward
        output = self.forward(x)

        # Calculating loss and accuracy
        loss = self.cross_entropy_loss(labels, output)
        accuracy = self.accuracy(output, labels)

        return loss, accuracy


# This class inherits from NeuralNetwork using 2 hidden layers

class TwoHiddenLayer(NeuralNetwork):
    def __init__(self, inputs, hidden1, hidden2, output):
        super().__init__()
        # Parameters and initializations
        # Before ReLU weights are multiplied by 2 since the half of its input is 0
        # Source: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        self.model = dict()
        self.model['W1'] = 2 * np.random.randn(inputs, hidden1) / np.sqrt(inputs)
        self.model['W2'] = 2 * np.random.randn(hidden1, hidden2) / np.sqrt(hidden1)
        self.model['W3'] = np.random.randn(hidden2, output) / np.sqrt(hidden2)

        # Class initializations
        self.out_activation1 = np.zeros((1, 1))
        self.out_activation2 = np.zeros((1, 1))

    def forward(self, x):
        # Forward propagation through our network
        out_product1 = np.dot(x, self.model['W1'])
        self.out_activation1 = self.relu(out_product1)

        out_product2 = np.dot(self.out_activation1, self.model['W2'])
        self.out_activation2 = self.relu(out_product2)

        out_product3 = np.dot(self.out_activation2, self.model['W3'])
        out_activation3 = self.stable_softmax(out_product3)

        return out_activation3

    def forward_propagation_with_dropout(self, x, keep_prob=0.5):
        # Implement Forward Propagation to calculate A2 (probabilities)
        out_product1 = np.dot(x, self.model['W1'])
        self.out_activation1 = self.relu(out_product1)

        # Dropout
        # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        d1 = np.random.rand(self.out_activation1.shape[0], self.out_activation1.shape[1])
        d1 = d1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
        self.out_activation1 = np.multiply(self.out_activation1, d1)
        self.out_activation1 = self.out_activation1/keep_prob

        out_product2 = np.dot(self.out_activation1, self.model['W2'])
        self.out_activation2 = self.relu(out_product2)

        out_product3 = np.dot(self.out_activation2, self.model['W3'])
        out_activation3 = self.stable_softmax(out_product3)

        return out_activation3, d1

    def backward(self, x, y, output, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = self.one_hot_cross_entropy_prime_with_softmax(y, output) / y.shape[0]

        hidden2_error = output_delta.dot(self.model['W3'].T)
        hidden2_delta = hidden2_error * self.relu_prime(self.out_activation2)

        hidden1_error = hidden2_delta.dot(self.model['W2'].T)
        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)

        self.model['W3'] -= (self.out_activation2.T.dot(output_delta)) * learning_rate
        self.model['W2'] -= (self.out_activation1.T.dot(hidden2_delta)) * learning_rate
        self.model['W1'] -= (x.T.dot(hidden1_delta)) * learning_rate

    def backward_propagation_with_dropout(self, x, y, output, d1, keep_prob, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = self.one_hot_cross_entropy_prime_with_softmax(y, output) / y.shape[0]

        hidden2_error = output_delta.dot(self.model['W3'].T)
        hidden2_delta = hidden2_error * self.relu_prime(self.out_activation2)
        # dropout
        hidden1_error = hidden2_delta.dot(self.model['W2'].T)
        # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        hidden1_error = np.multiply(d1, hidden1_error)
        # Step 2: Scale the value of neurons that haven't been shut down
        hidden1_error = hidden1_error / keep_prob

        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)
        # reload w
        self.model['W3'] -= (self.out_activation2.T.dot(output_delta)) * learning_rate
        self.model['W2'] -= (self.out_activation1.T.dot(hidden2_delta)) * learning_rate
        self.model['W1'] -= (x.T.dot(hidden1_delta)) * learning_rate

    def feed_backward(self, y):
        # Forward propagation through our network
        out_product1 = np.dot(y, self.model['W3'].T)

        out_activation1 = self.relu(out_product1)
        out_product2 = np.dot(out_activation1, self.model['W2'].T)

        out_activation2 = self.relu(out_product2)
        out_product3 = np.dot(out_activation2, self.model['W1'].T)

        return out_product3


# This class inherits from NeuralNetwork using 1 hidden layers

class OneHiddenLayer(NeuralNetwork):
    def __init__(self, inputs, hidden1, output):
        super().__init__()
        # Parameters and initializations
        # Before ReLU weights are multiplied by 2 since the half of its input is 0
        # Source: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        self.model = dict()
        self.model['W1'] = np.random.randn(inputs, hidden1) / np.sqrt(inputs)
        self.model['W2'] = np.random.randn(hidden1, output) / np.sqrt(hidden1)

        # Class initializations
        self.out_activation1 = np.zeros((1, 1))

    def forward(self, x):
        # Forward propagation through our network
        out_product1 = np.dot(x, self.model['W1'])
        self.out_activation1 = self.relu(out_product1)

        out_product3 = np.dot(self.out_activation1, self.model['W2'])
        out_activation3 = self.stable_softmax(out_product3)

        return out_activation3

    def forward_propagation_with_dropout(self, x, keep_prob=0.5):
        # Implement Forward Propagation to calculate A2 (probabilities)
        out_product1 = np.dot(x, self.model['W1'])
        self.out_activation1 = self.relu(out_product1)

        # Dropout
        # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        d1 = np.random.rand(self.out_activation1.shape[0], self.out_activation1.shape[1])
        d1 = d1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
        self.out_activation1 = np.multiply(self.out_activation1, d1)
        self.out_activation1 = self.out_activation1/keep_prob

        out_product2 = np.dot(self.out_activation1, self.model['W2'])
        out_activation2 = self.stable_softmax(out_product2)

        return out_activation2, d1

    def backward(self, x, y, output, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = self.one_hot_cross_entropy_prime_with_softmax(y, output) / y.shape[0]

        hidden1_error = output_delta.dot(self.model['W2'].T)
        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)

        self.model['W2'] -= (self.out_activation1.T.dot(output_delta)) * learning_rate
        self.model['W1'] -= (x.T.dot(hidden1_delta)) * learning_rate

    def backward_propagation_with_dropout(self, x, y, output, d1, keep_prob, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = self.one_hot_cross_entropy_prime_with_softmax(y, output) / y.shape[0]

        # dropout
        hidden1_error = output_delta.dot(self.model['W2'].T)
        # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        hidden1_error = np.multiply(d1, hidden1_error)
        # Step 2: Scale the value of neurons that haven't been shut down
        hidden1_error = hidden1_error / keep_prob

        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)
        # reload w
        self.model['W2'] -= (self.out_activation1.T.dot(output_delta)) * learning_rate
        self.model['W1'] -= (x.T.dot(hidden1_delta)) * learning_rate

    def feed_backward(self, y):
        # Forward propagation through our network
        out_product1 = np.dot(y, self.model['W2'].T)

        out_activation1 = self.relu(out_product1)
        out_product2 = np.dot(out_activation1, self.model['W1'].T)

        return out_product2
