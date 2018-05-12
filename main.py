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
        # Before ReLU weights are multiplied by 2 since the half of its input is 0
        # Source: http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        self.model = dict()
        self.model['W1'] = 2 * np.random.randn(inputs, hidden1) / np.sqrt(inputs)
        self.model['W2'] = 2 * np.random.randn(hidden1, hidden2) / np.sqrt(hidden1)
        self.model['W3'] = np.random.randn(hidden2, output) / np.sqrt(hidden2)

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
        output_delta = y * self.one_hot_cross_entropy_prime_with_softmax(y, output)

        hidden2_error = output_delta.dot(self.model['W3'].T)
        hidden2_delta = hidden2_error * self.relu_prime(self.out_activation2)

        hidden1_error = hidden2_delta.dot(self.model['W2'].T)
        hidden1_delta = hidden1_error * self.relu_prime(self.out_activation1)

        # Este print imprime los Ws antes de actualizarlos
        # En la segunda iteracion ya son nans
        print(self.model["W1"])
        print(self.model["W2"])
        print(self.model["W3"])

        self.model['W3'] = self.out_activation2.T.dot(output_delta) * learning_rate
        self.model['W2'] = self.out_activation1.T.dot(hidden2_delta) * learning_rate
        self.model['W1'] = x.T.dot(hidden1_delta) * learning_rate

    def backward_propagation_with_dropout(self, x, y, output, d1, keep_prob, learning_rate=0.0085):
        # y is a one_hot_vector
        output_delta = y * self.one_hot_cross_entropy_prime_with_softmax(y, output)

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

    def train(self, x, y, batch_size, epoch):

        # Converting labels to one-hot encoding
        labels = self.to_one_hot(y)

        # Calculating batch size
        total = x.shape[0]
        batches = total / batch_size

        # Dividing by mini-batches
        batches_data = np.split(x, batches, axis=0)
        batches_labels = np.split(labels, batches, axis=0)

        # For each epoch
        for i in range(epoch):
            print("Epoch #", i)
            # Take each mini-batch and train
            for batch_data, batch_labels in zip(batches_data, batches_labels):
                # Jocelyn... Cuando lo corra y vea nans detengalo y vayase arriba.
                # El primer output tiene resultados, los demás son nan
                print(batch_data.shape)
                print(batch_labels.shape)
                output = self.forward(batch_data)
                print(output)
                # Vaya a esta funcion y revise los prints que puse
                self.backward(batch_data, batch_labels, output, learning_rate=0.0085)


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
    print("Loading MNIST data set...")
    data = MNIST("./MNIST_data_set")
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    # Converting to numpy arrays
    print("Preparing data...")
    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Getting dimensions
    first_layer = training_images.shape[1]
    last_layer = training_labels.max() + 1

    # Creating neural network
    print("Initializing neural network...")
    neural_network = NeuralNetwork(first_layer, 512, 512, last_layer)

    # WORKING ON...
    print("WORKING ON... TRAINING...")
    neural_network.train(training_images, training_labels, 32, 1)


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

    # print(NeuralNetwork.forward(a))

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
    # test()
