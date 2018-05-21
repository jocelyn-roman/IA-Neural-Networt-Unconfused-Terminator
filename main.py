"""
Created on Apr 27, 2018

@author: J&J
"""

from mnist import MNIST
import numpy as np
import neural_network as nn
import utils as utl


def main():
    # Loading MNIST data set
    print("Loading MNIST data set...")
    data = MNIST("./MNIST_data_set")
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    # Converting to numpy arrays and normalizing images
    print("Preparing data...")
    training_images = np.array(training_images) / 255
    training_labels = np.array(training_labels)
    test_images = np.array(test_images) / 255
    test_labels = np.array(test_labels)

    train = dict()
    train['images'] = training_images
    train['labels'] = training_labels

    test = dict()
    test['images'] = test_images
    test['labels'] = test_labels

    # Getting dimensions
    first_layer = training_images.shape[1]
    last_layer = training_labels.max() + 1

    # Creating neural network
    print("ONE HIDDEN LAYER")
    print("128")
    neural_network = nn.OneHiddenLayer(first_layer, 128, last_layer)
    train_network(neural_network, train, "one_128")
    test_data(neural_network, test)

    print("256")
    neural_network = nn.OneHiddenLayer(first_layer, 256, last_layer)
    train_network(neural_network, train, "one_256")
    test_data(neural_network, test)

    print("512")
    neural_network = nn.OneHiddenLayer(first_layer, 512, last_layer)
    train_network(neural_network, train, "one_512")
    test_data(neural_network, test)

    print("1024")
    neural_network = nn.OneHiddenLayer(first_layer, 1024, last_layer)
    train_network(neural_network, train, "one_1024")
    test_data(neural_network, test)

    print("2048")
    neural_network = nn.OneHiddenLayer(first_layer, 2048, last_layer)
    train_network(neural_network, train, "one_2048")
    test_data(neural_network, test)

    print("TWO HIDDEN LAYERS")
    print("128")
    neural_network = nn.TwoHiddenLayer(first_layer, 128, 128, last_layer)
    train_network(neural_network, train, "two_128")
    test_data(neural_network, test)

    print("256")
    neural_network = nn.TwoHiddenLayer(first_layer, 256, 256, last_layer)
    train_network(neural_network, train, "two_256")
    test_data(neural_network, test)

    print("512")
    neural_network = nn.TwoHiddenLayer(first_layer, 512, 512, last_layer)
    train_network(neural_network, train, "two_512")
    test_data(neural_network, test)

    print("1024")
    neural_network = nn.TwoHiddenLayer(first_layer, 1024, 1024, last_layer)
    train_network(neural_network, train, "two_1024")
    test_data(neural_network, test)

    print("2048")
    neural_network = nn.TwoHiddenLayer(first_layer, 2048, 2048, last_layer)
    train_network(neural_network, train, "two_2048")
    test_data(neural_network, test)


def train_network(network, data, weights_file):
    # Training neural network
    print("Training...")
    network.train(data['images'], data['labels'], 32, 4)
    network.plot("network_"+weights_file)

    # Saving weights into file
    print("Saving weights")
    network.save("output/weights/network_"+weights_file+".pickle")


def test_data(network, data):
    # Testing the neural network with the test data set
    print("Testing...")
    loss, accuracy = network.test(data['images'], data['labels'])
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


def test_custom_numbers():
    network = nn.OneHiddenLayer(784, 128, 10)
    network.load("output/weights/network_one_128.pickle")

    print("Testing with a local image")
    image = utl.load_image("Test_data/zero_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/one_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/two_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/three_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/four_2.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/five_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/six_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/seven_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/eight_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    # utl.plot_probability(probability[0])

    image = utl.load_image("Test_data/nine_1.png")
    utl.visualize_image(image, "TEST")
    probability = network.forward(image)
    print(np.argmax(probability))
    utl.plot_probability(probability[0])


def test_feed_backward():
    # Setting up neural network
    network = nn.OneHiddenLayer(784, 128, 10)
    network.load("output/weights/network_one_128.pickle")

    labels = np.array([[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]])

    images = network.feed_backward(labels) * 255

    utl.visualize_image(images[0], "Zero")
    utl.visualize_image(images[1], "One")
    utl.visualize_image(images[2], "Two")
    utl.visualize_image(images[3], "Three")
    utl.visualize_image(images[4], "Four")
    utl.visualize_image(images[5], "Five")
    utl.visualize_image(images[6], "Six")
    utl.visualize_image(images[7], "Seven")
    utl.visualize_image(images[8], "Eight")
    utl.visualize_image(images[9], "Nine")


if __name__ == "__main__":
    # main()
    test_custom_numbers()
    # test_feed_backward()
